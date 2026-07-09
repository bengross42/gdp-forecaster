# %%
## Importing necessary packages

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
import os

# %%
def construct_var_matrix(Y, p, X_exog=None):
    Y = np.asarray(Y)
     
    T, n = Y.shape
    k_endog = 1 + n * p
    
   
    X_endog = np.zeros((T - p, k_endog))
    Y_dep = Y[p:]
    for t in range(T - p):
        
        X_endog[t, 0] = 1.0
        for lag in range(1, p + 1):
            X_endog[t, 1 + (lag-1)*n : 1 + lag*n] = Y[p + t - lag]
            
    # --- NEW: Add exogenous variables ---
    
    if X_exog is not None:
        X_exog=np.asarray(X_exog)
        # X_exog should be (T x r). We slice it to match the dependent variable.
        X_exog_trimmed = X_exog[p:] 
        X = np.column_stack((X_endog, X_exog_trimmed))
    else:
        
        X = X_endog

    return Y_dep, X

def process_data(df, new_cols, QoQ = True):

    df.columns = ["Quarter", "HKGDP", "HKGDP_yoy", "Imports", "Exports", "RSV", "HSI", "PPI", "PST_Volume", "FFR", "China_PMI_NEO", "CCPI"]

    cols = ["HKGDP", "HKGDP_yoy", "Imports", "Exports", "RSV", "HSI", "PPI", "PST_Volume", "FFR", "China_PMI_NEO", "CCPI"]
    
    df=df.drop(index=0)

    ## Changing data to correct types

    df['Quarter'] = pd.to_datetime(df['Quarter'], format="%m/%Y", errors="coerce")
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.set_index('Quarter')

    ## ADJUSTING IMPORTS, EXPORTS, and RETAIL SALES VALUE FOR INFLATION AND MAKING THEM YOY%

    df["Imports_adj"] = df["Imports"] / df["CCPI"]
    df["Exports_adj"] = df["Exports"] / df["CCPI"]
    df["RSV_adj"] = df["RSV"] / df["CCPI"]

    ## Taking YOY% growth for each (INCLUDING China PMI_NEO)

    df["Imports"] = (df["Imports_adj"] - df["Imports_adj"].shift(4))/df["Imports_adj"].shift(4)
    df["Exports"] = (df["Exports_adj"] - df["Exports_adj"].shift(4))/df["Exports_adj"].shift(4)
    df["RSV"] = (df["RSV_adj"] - df["RSV_adj"].shift(4))/df["RSV_adj"].shift(4)
    df["China_PMI_NEO"] = (df["China_PMI_NEO"] - df["China_PMI_NEO"].shift(4))/df["China_PMI_NEO"].shift(4)

    df=df.drop(columns = ["Imports_adj", "Exports_adj", "RSV_adj"])

    ## Annualizing Quarterly GDP

    df["HKGDP"] = ((1+(df["HKGDP"])/100)**4 - 1)

    ## Taking log difference of Hang Seng Index

    df["HSI_log"] = np.log(df["HSI"])
    df["HSI"] = df["HSI_log"].diff()

    ## Taking first difference of Federal Funds Rate

    df["FFR"] = df["FFR"].diff()

    ## Dividing select variables by 100 to get relatively consistent orders of magnitude

    df["PPI"] = df["PPI"]/100
    df["PST_Volume"] = df["PST_Volume"]/100


    df = df.drop(columns = ["HSI_log", "CCPI"])

    # Get GDP in comparable units as before

    df = df*100

    if QoQ:
        ### Option to use HKGDP QoQ

        df=df.drop(columns = "HKGDP_yoy")
    else:
        ### Option to use HKGDP YoY

        df["HKGDP"] = df["HKGDP_yoy"]/100
        df=df.drop(columns = "HKGDP_yoy")

    df = clean_timeseries_dataset(df, new_cols)

    #==============================================#
    ## ADDING A COVID DUMMY FOR COMPARISON
    #==============================================#

    # Define COVID period (adjust as needed)

    covid_start = '2020Q1'
    covid_end = '2021Q2'

    df["Covid"] = ((df.index < covid_end) & (df.index > covid_start)) 
    df['Covid'] = df['Covid'].astype(int)

    #==============================================#
    ## ADDING A GLOBAL FINANCIAL CRISIS DUMMY
    #==============================================#

    # Define GFC period (adjust as needed)

    gfc_start = '2008Q3'
    gfc_end = '2009Q2'

    df["GFC"] = ((df.index < gfc_end) & (df.index > gfc_start)) 
    df['GFC'] = df['GFC'].astype(int)

    return df


# %%
def minnesota_prior(Y, p, X_exog=None, lambda_val=0.2, delta=0.5, decay=2, exog_dict=None, constant_var = 1e6):
    #"""Construct Minnesota prior for BVAR."""
    # Exogeneity_dict is read as: key is exogenous to values
    # If USGDP is index 3 and is exogenous to HKGDP (index 0), then we have {3: [0]}
    Y = np.asarray(Y) 
    

    r=0
    if X_exog is not None:
        r = X_exog.shape[1]

    X_exog=np.asarray(X_exog) 
    T, n = Y.shape
    k_endog = 1 + n * p
    k_total = k_endog + r
    k = k_total

    # --- Handle Exogeneity Soft Restrictions ---
    if exog_dict is None:
        exog_dict = {}

    # Prior means: random walk for own first lag, zero otherwise
    # B_prior contains (1+n*p coefficients for each of n equations)
    B_prior = np.zeros((k, n))
    for i in range(n):
        B_prior[1 + i, i] = 1.0 # own first lag = 1 

    ## Sigmas contain estimated standard errors for each variable's AR(p) model
    sigmas = np.zeros(n)

    for i in range(n):
        y_i = Y[:, i]
    
        # Construct X matrix for this specific variable's AR(p) model
        # Column 1 is the constant, columns 2 to p+1 are the lags
        X_i = np.column_stack([np.ones(T - p)] + [y_i[p - lag:T - lag] for lag in range(1, p + 1)])
    
        # Simple OLS: beta = (X'X)^-1 X'Y
        beta_i = np.linalg.lstsq(X_i, y_i[p:], rcond=None)[0]
    
        # Calculate residuals
        resid_i = y_i[p:] - X_i @ beta_i
    
        # MLE variance calculation
        sigma2_manual = np.mean(resid_i**2)
        sigmas[i] = sigma2_manual

    #Prior covariance for vec(B): diagonal
    # Order: for each equation i, coefficients [const, lag1_var1, ..., lag1_varn, lag2_var1, ...]
    ## V_prior contains all covariances, so it must be (# of coefficients x # of coefficients)
    V_prior = np.zeros((k * n, k * n))

    # For each of the 'n' equations, set its constant variance to a massive number.
    # This step may seem unnecessary with standardized data, but we keep it in order to simplify code
    # and allow for constant terms in subsampling. Also, OLS should return ~0 constants anyway.
    for i in range(n):
        row_idx_const = i * k + 0
        V_prior[row_idx_const, row_idx_const] = constant_var  # 1,000,000 is effectively "no prior opinion"

    ## Iterates through each equation-lag-coefficient triplet, assigning variances (NOT covariances)    
    for i in range(n): # equation
        for lag in range(1, p + 1):
            for j in range(n): # variable
                col_idx = 1 + (lag - 1) * n + j # (Variable-lag combo)
                row_idx = i * k + col_idx # ()
                if i == j:
                    var_ij = (lambda_val ** 2) / (lag ** decay)
                else:
                    var_ij = (lambda_val ** 2) / (lag ** decay) * (delta ** 2) * (sigmas[i] ** 2) / (sigmas[j] ** 2)
                
                # --- NEW: SOFT EXOGENEITY RESTRICTION ---
                # If variable j is listed in the exog_dict, check if it is 
                # restricted from impacting variable i.
                if j in exog_dict and i in exog_dict[j]:
                    var_ij = 1e-8  # Crush the variance to near-zero

                # Ensure no zeros to prevent singular matrix
                #V_prior[row_idx, row_idx] = max(var_ij, 1e-8)
                V_prior[row_idx, row_idx] = var_ij 

     # 2. NEW: Diffuse Prior for the Exogenous block
    for i in range(n): # For each equation
        for exog_col in range(r): # For each dummy variable
            # The column index in the total X matrix is k_endog + exog_col
            col_idx = k_endog + exog_col
            row_idx = i * k + col_idx
            
            # Set a massive prior variance so OLS completely determines the dummy's effect
            V_prior[row_idx, row_idx] = 1e6 


    # Prior for Sigma: diagonal with sigmas^2 on diagonal
    # Note that our sigma prior, being diagonal, implies that there are no correlated errors between
    # equations within the same time period. Error variances are assumed to be equal to the errors
    # estimated in an AR(p) univariate process for each variable. This process is allowed to update
    # with no friction, unlike B and V!
    S_prior = np.diag(sigmas ** 2)
    nu_prior = n + 2 # minimal degrees of freedom
    return B_prior, V_prior, S_prior, nu_prior

# %%
def estimate_bvar(Y, p, X_exog = None, lambda_val=0.2, delta=0.5, decay=2, n_draws=2000, exog_dict = None, constant_var = 1e6, 
                  plot_var_idx = None, print_eq_idx = None, var_names = None, exog_names = None):
    ##Estimate BVAR with Minnesota prior (Normal-Inverse-Wishart)
    
    T, n = Y.shape

    r=0
    if X_exog is not None:
        r = X_exog.shape[1]

    k_endog = 1 + n * p
    k_total = k_endog + r
    k = k_total

    Y_mat, X_mat = construct_var_matrix(Y, p, X_exog)
    T_eff = Y_mat.shape[0] 

    B_ols = np.linalg.lstsq(X_mat, Y_mat, rcond=None)[0]
    resid = Y_mat - X_mat @ B_ols
    S_ols = resid.T @ resid / T_eff 

    B_prior, V_prior, S_prior, nu_prior = minnesota_prior(
    Y, p, X_exog, lambda_val, delta, decay, exog_dict, constant_var
    ) 
    
    V_post_inv = np.linalg.inv(V_prior) + np.kron(np.eye(n), X_mat.T @ X_mat)
    V_post = np.linalg.inv(V_post_inv)

    vec_B_prior = B_prior.T.flatten()
    vec_B_ols = B_ols.T.flatten()
    vec_B_post = V_post @ (np.linalg.inv(V_prior) @ vec_B_prior + np.kron(np.eye(n), X_mat.T) @ Y_mat.T.flatten())
    B_post = vec_B_post.reshape(n, k).T 

    nu_post = nu_prior + T_eff
    S_post = S_prior + resid.T @ resid + (B_ols - B_post).T @ (X_mat.T @ X_mat) @ (B_ols - B_post) 

    B_draws = np.zeros((n_draws, k, n))
    Sigma_draws = np.zeros((n_draws, n, n))
    L_V = cholesky(V_post, lower=True) 
    df = nu_post
    L_S = cholesky(S_post, lower=True)

    for s in range(n_draws):
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = np.sqrt(np.random.chisquare(df - i))
            for j in range(i):
                A[i, j] = np.random.randn()
        W = L_S @ A @ A.T @ L_S.T
        Sigma_draws[s] = np.linalg.inv(W)

        vec_B_draw = vec_B_post + L_V @ np.random.randn(k * n)
        B_draws[s] = vec_B_draw.reshape(n,k).T

    # ==========================================
    # PLOTTING LOGIC (Distinct Vertical Bars)
    # ==========================================
    if plot_var_idx is not None:
        coeff_draws = B_draws[:, :, plot_var_idx]
        
        median = np.median(coeff_draws, axis=0)
        lower_95 = np.percentile(coeff_draws, 2.5, axis=0)
        upper_95 = np.percentile(coeff_draws, 97.5, axis=0)
        lower_68 = np.percentile(coeff_draws, 16, axis=0)
        upper_68 = np.percentile(coeff_draws, 84, axis=0)
        
        x_labels = ["Const"]
        if var_names is None:
            var_names = [f"Var{i}" for i in range(n)]
            
        for lag in range(1, p + 1):
            for name in var_names:
                x_labels.append(f"{name}_L{lag}")
                
        if exog_names is None and X_exog is not None:
            exog_names = [f"Exog{i}" for i in range(r)]
            
        if exog_names is not None:
            x_labels.extend(exog_names)
            
        fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.8), 5))
        x_axis = np.arange(len(x_labels))
        
        bar_width = 0.6 
        
        ax.bar(x_axis, upper_95 - lower_95, width=bar_width, bottom=lower_95, 
               color='blue', alpha=0.2, label='95% CI', edgecolor='none')
        ax.bar(x_axis, upper_68 - lower_68, width=bar_width, bottom=lower_68, 
               color='blue', alpha=0.4, label='68% CI', edgecolor='none')
        ax.plot(x_axis, median, color='black', marker='o', linestyle='None', 
                markersize=5, label='Median')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels, rotation=90)
        
        eq_name = var_names[plot_var_idx] if var_names else f"Variable {plot_var_idx}"
        ax.set_title(f"Posterior Coefficients for Equation: {eq_name}")
        ax.set_ylabel("Coefficient Value")
        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.legend()
        plt.tight_layout()
        #plt.show()

        # ==========================================
    # NEW: EQUATION PRINTING LOGIC
    # ==========================================
    if print_eq_idx is not None:
        # Added 't' to the translation map to subscript the 't' as well
        sub_map = str.maketrans("t-0123456789", "ₜ₋₀₁₂₃₄₅₆₇₈₉")
        def to_sub(text): 
            return str(text).translate(sub_map)

        if var_names is None:
            var_names = [f"Var{i}" for i in range(n)]
            
        # Subscript the dependent variable's 't'
        dep_name = var_names[print_eq_idx]
        coeffs = B_post[:, print_eq_idx]
        
        # 1. Start the equation string
        c_val = coeffs[0]
        if c_val >= 0:
            eq_str = f"{dep_name}{to_sub('t')} =  {c_val:.4f} "
        else:
            eq_str = f"{dep_name}{to_sub('t')} = -{abs(c_val):.4f} "
            
        # 2. Add lagged endogenous variables (passing "t-lag" to the subscript function)
        for lag in range(1, p + 1):
            lag_terms = []
            for j in range(n):
                idx = 1 + (lag - 1) * n + j
                val = coeffs[idx]
                name = var_names[j]
                
                if val >= 0:
                    lag_terms.append(f"+ {val:.4f} {name}{to_sub(f't-{lag}')}")
                else:
                    lag_terms.append(f"- {abs(val):.4f} {name}{to_sub(f't-{lag}')}")
                    
            indent = " " * (len(dep_name) + 4)
            eq_str += "\n" + indent + " ".join(lag_terms)
            
        # 3. Add exogenous variables (subscripting just the 't')
        if exog_names is None and X_exog is not None:
            exog_names = [f"Exog{i}" for i in range(r)]
            
        if exog_names is not None:
            exog_terms = []
            for m in range(r):
                idx = 1 + n * p + m
                val = coeffs[idx]
                name = exog_names[m]
                if val >= 0:
                    exog_terms.append(f"+ {val:.4f} {name}{to_sub('t')}")
                else:
                    exog_terms.append(f"- {abs(val):.4f} {name}{to_sub('t')}")
                    
            indent = " " * (len(dep_name) + 4)
            eq_str += "\n" + indent + " ".join(exog_terms)
            
        # 4. Add error term
        indent = " " * (len(dep_name) + 4)
        eq_str += "\n" + indent + f"+ ε{to_sub('t')}"
        
        # Print to console
        print("\n" + "="*60)
        print(" ESTIMATED EQUATION (Posterior Mean Coefficients - All Variables Standardized):")
        print("="*60)
        print(eq_str)
        print("="*60 + "\n")
    # ==========================================
    return B_draws, Sigma_draws, B_post, S_post

# %%
def unconditional_forecast(Y, p, B_draws, Sigma_draws, h_steps, future_exog = None):
    """
    Unconditional forecast: Simulate all variables forward freely.
    
    Parameters:
    -----------
    Y : T x n data
    p : lag length
    B_draws : S x k x n posterior draws
    Sigma_draws : S x n x n posterior draws
    h_steps : number of forecast steps

    Returns:
    --------
    forecast_draws : S x h_steps x n array of forecast draws
    """

    Y = np.asarray(Y)
    T, n = Y.shape
    S = B_draws.shape[0]
    k_total = B_draws.shape[1]     # Total columns in B (Constant + Lags + Dummies)
    k_endog = 1 + n * p           # Columns that belong to standard VAR lags
    r = k_total - k_endog         # Infer number of dummy variables automatically!

    # Safely force future_exog into a 2D array of shape (h_steps, r)
    if future_exog is not None:
        future_exog = np.asarray(future_exog).reshape(h_steps, -1)
        if future_exog.shape[1] != r:
            raise ValueError(f"future_exog has {future_exog.shape[1]} columns, but B_draws implies {r} exogenous variables.")

    forecast_draws = np.zeros((S, h_steps, n))

    # Initialize with last p observations
    y_hist = Y[-p:].copy() # p x n

    # OUTSIDE LOOP: Parallel Universes (Preserves true variance)
    for s in range(S):
        B = B_draws[s] 
        Sigma = Sigma_draws[s] 
        L_Sigma = cholesky(Sigma, lower=True)
        
        # Private history for this specific universe
        y_hist = Y[-p:].copy() 
        
        # INSIDE LOOP: Time steps
        for h in range(h_steps):
            # Construct X for this step
            x_t = np.zeros(k_total) # Use total size
            x_t[0] = 1.0
            for lag in range(1, p + 1):
                x_t[1 + (lag-1)*n : 1 + lag*n] = y_hist[-lag]

            # Append future dummies (if any) to the end of X
            if future_exog is not None: 
                x_t[k_endog:] = future_exog[h] # Safe because future_exog is now 2D

            # Draw all n variables simultaneously
            y_mean = x_t @ B 
            y_t = y_mean + L_Sigma @ np.random.randn(n)

            # Save the draw
            forecast_draws[s, h, :] = y_t

            # Update THIS universe's private history
            y_hist = np.vstack([y_hist, y_t])

    return forecast_draws

# %%
def dynamic_conditional_forecast(Y, p, B_draws, Sigma_draws, h_steps, future_exog = None, condition_dict = False):
    """
    Advanced Conditional Forecast with differing condition lengths.
    Simulates full coherent paths (Parallel Universes) to preserve true variance.
    
    Parameters:
    -----------
    Y : T x n data
    p : lag length
    B_draws : S x k x n posterior draws
    Sigma_draws : S x n x n posterior draws
    h_steps : TOTAL number of forecast steps (e.g., 6)
    condition_dict : dict 
        Format: {var_idx: np.array([val_h0, val_h1, ...])}
        Example: {9: np.array([1.5, 1.8]), 3: np.array([2.0])}
        This means US GDP (idx 9) is locked for 2 steps, CPI (idx 3) locked for 1 step.
        All other variables are completely free for all 6 steps.

    Returns:
    --------
    forecast_draws : S x h_steps x n array of forecast draws
    """
    
    
    # Handle the case where no conditions are passed at all (pure unconditional)
    if not condition_dict:
        condition_dict = {}
        forecast_draws = unconditional_forecast(Y, p, B_draws, Sigma_draws, h_steps, future_exog = None)
        return forecast_draws
    
    Y = np.asarray(Y)
    T, n = Y.shape
    S = B_draws.shape[0]
    k_total = B_draws.shape[1]     # Total columns in B (Constant + Lags + Dummies)
    k_endog = 1 + n * p           # Columns that belong to standard VAR lags
    r = k_total - k_endog         # Infer number of dummy variables automatically!

    # Safely force future_exog into a 2D array of shape (h_steps, r)
    if future_exog is not None:
        future_exog = np.asarray(future_exog).reshape(h_steps, -1)
        if future_exog.shape[1] != r:
            raise ValueError(f"future_exog has {future_exog.shape[1]} columns, but B_draws implies {r} exogenous variables.")

    forecast_draws = np.zeros((S, h_steps, n))

    # OUTSIDE LOOP: Parallel Universes
    for s in range(S):
        B = B_draws[s] 
        Sigma = Sigma_draws[s] 
        
        # INSIDE LOOP: Private history for this draw
        y_hist = Y[-p:].copy() 
        
        # INSIDE LOOP: Iterate over total time steps (e.g., 0 to 5)
        for h in range(h_steps):
            
            # 1. DYNAMICALLY determine what is conditioned THIS step
            # A variable is conditioned at step h ONLY if it's in the dictionary 
            # AND we haven't run out of pre-supplied values for it.
            cond_idx_h = [idx for idx in condition_dict.keys() if h < len(condition_dict[idx])]
            free_idx_h = [i for i in range(n) if i not in cond_idx_h]

            # 2. Construct X for this step
            x_t = np.zeros(k_total)
            x_t[0] = 1.0
            for lag in range(1, p + 1):
                x_t[1 + (lag-1)*n : 1 + lag*n] = y_hist[-lag]

            # Append future dummies (if any) to the end of X
            if future_exog is not None: 
                x_t[k_endog:] = future_exog[h] # Safe because future_exog is now 2D

            y_mean = x_t @ B 
            y_var = Sigma 

            # 3. Branching Logic: Are we conditioning on anything this step?
            if len(cond_idx_h) == 0:
                # --- CASE A: PURELY UNCONDITIONAL STEP ---
                # All variables are free. Draw from the full multivariate normal.
                L_Sigma = cholesky(y_var, lower=True)
                y_t = y_mean + L_Sigma @ np.random.randn(n)
            
            else:
                # --- CASE B: PARTIAL CONDITIONING STEP ---
                # Some variables are locked, some are free. Use the partition math.
                mu_c = y_mean[cond_idx_h]
                mu_f = y_mean[free_idx_h]
                
                Sigma_ff = y_var[np.ix_(free_idx_h, free_idx_h)]
                Sigma_cc = y_var[np.ix_(cond_idx_h, cond_idx_h)]
                Sigma_fc = y_var[np.ix_(free_idx_h, cond_idx_h)]

                Sigma_cc_inv = np.linalg.inv(Sigma_cc)

                # Extract the specific values for the variables locked at this step h
                cond_vals_h = np.array([condition_dict[idx][h] for idx in cond_idx_h])

                mu_star = mu_f + Sigma_fc @ Sigma_cc_inv @ (cond_vals_h - mu_c)
                Sigma_star = Sigma_ff - Sigma_fc @ Sigma_cc_inv @ Sigma_fc.T

                # Ensure symmetry and positive definiteness
                Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
                eigvals = np.linalg.eigvalsh(Sigma_star)
                if np.any(eigvals <= 0):
                    Sigma_star += (abs(min(eigvals)) + 1e-6) * np.eye(len(free_idx_h))

                L_star = cholesky(Sigma_star, lower=True)
                y_free_draw = mu_star + L_star @ np.random.randn(len(free_idx_h))

                # Assemble full y_t
                y_t = np.zeros(n)
                y_t[free_idx_h] = y_free_draw
                y_t[cond_idx_h] = cond_vals_h

            # Save the draw
            forecast_draws[s, h, :] = y_t

            # Update THIS draw's private history
            y_hist = np.vstack([y_hist, y_t])

    return forecast_draws

# %%
def forecast_graph(forecast_draws, actual_HKGDP, test_dates, p, lambda_val, delta, 
                   decay, standardization_dict, is_diff=False, last_train_value=None, include_training=False, train = False, isplot = True):
    """
    Calculates RMSE, prints results, and plots forecasts.
    Robust to both Levels and First-Differenced HKGDP.
    
    Parameters:
    -----------
    forecast_draws : S x h_steps x n array (Standardized draws)
    actual_HKGDP : array-like (Raw actual GDP for the test set)
    is_diff : bool (True if forecast_draws are for 1st differences)
    Y_means : array (Means from the training set used for standardization)
    Y_stds : array (Stds from the training set used for standardization)
    last_train_value : float (The RAW level of HKGDP at the end of the training set. 
                                 Only used if is_diff=True)
    test_dates : array-like (Dates for the x-axis of the plot)
    """
    
    actual_HKGDP = np.asarray(actual_HKGDP)

    if forecast_draws.shape[1] < len(actual_HKGDP):
        actual_HKGDP = actual_HKGDP[:forecast_draws.shape[1]]
        test_dates = test_dates[:forecast_draws.shape[1]]
        print(actual_HKGDP)
        print(test_dates)
        print("=========================")
    

    if not is_diff:
        mean_scale = standardization_dict["HKGDP"][0]
        std_scale = standardization_dict["HKGDP"][1]
        # --- LEVELS / STANDARDIZED GROWTH RATES ---
        # 1. Un-standardize the draws
        unstd_draws = forecast_draws[:, :, 0] * std_scale + mean_scale
        
        # 2. Calculate median and bands
        median_forecast_path = np.median(unstd_draws, axis=0)
        lo_band = np.percentile(unstd_draws, 16, axis=0)
        hi_band = np.percentile(unstd_draws, 84, axis=0)

    else:
        mean_scale = standardization_dict["HKGDP_diff"][0]
        std_scale = standardization_dict["HKGDP_diff"][1]
        # --- FIRST DIFFERENCES ---
        S, h_steps, n = forecast_draws.shape
        level_draws = np.zeros((S, h_steps))
        
        # 1. Un-standardize the DIFFERENCES first!
        unstd_diff_draws = forecast_draws[:, :, 0] * std_scale + mean_scale
        
        # 2. Cumulatively sum the RAW differences to get RAW levels
        level_draws[:, 0] = last_train_value + unstd_diff_draws[:, 0]
        for i in range(1, h_steps):
            level_draws[:, i] = level_draws[:, i-1] + unstd_diff_draws[:, i]

        # 3. Calculate median and bands from the reconstructed levels
        median_forecast_path = np.median(level_draws, axis=0)
        lo_band = np.percentile(level_draws, 16, axis=0)
        hi_band = np.percentile(level_draws, 84, axis=0)
        print("Differenced HKGDP Detected....")

    # ---------------------------------------------------------
    # CALCULATE & PRINT RMSE
    # ---------------------------------------------------------
    final_rmse = np.sqrt(mean_squared_error(actual_HKGDP, median_forecast_path))

    if not(isplot):
        return final_rmse

    print(f"Actual HK GDP (Test Set):\n{actual_HKGDP}")
    print(f"\nMedian Forecast Path (Raw Levels):\n{np.round(median_forecast_path, 2)}")
    print(f"\nFinal Out-of-Sample RMSE: {final_rmse:.4f}")

    # ---------------------------------------------------------
    # PLOT THE FORECAST
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    
    # Plot median forecast
    plt.plot(test_dates, median_forecast_path, 'b-', linewidth=2, label='BVAR Median Forecast')
    
    # Plot 68% Confidence Band (16th to 84th percentile)
    plt.fill_between(test_dates, lo_band, hi_band, color='blue', alpha=0.2, label='68% Credible Interval')

    if include_training:
        plt.plot(train.index[40:], train['HKGDP'][40:], label='Train', color='C0')
        # Plot actuals
        plt.plot(test_dates, actual_HKGDP, 'k--', linewidth=2, label='Actual HK GDP', color = 'C1')
    else:
        # Plot actuals
        plt.plot(test_dates, actual_HKGDP, 'k--', linewidth=2, label='Actual HK GDP', marker='o')
    
    plt.title(f"HK GDP BVAR({p}, {lambda_val}, {delta}, {decay}) Forecast (RMSE: {final_rmse:.2f})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("HK GDP Growth (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()


# %%
def build_condition_dict(test, cond_setup, standardization_dict):
    condition_dict = {}
    for key in cond_setup.keys():
        index = test.columns.get_loc(key)
        mean_val = standardization_dict[key][0]
        std_val = standardization_dict[key][1]
        n_periods_ahead = cond_setup[key]
        condition_array = (test[key][0:n_periods_ahead] - mean_val) / std_val
        condition_dict[index] = condition_array.tolist()

    
    return condition_dict

# %%



def plot_pure_forecast(forecast_draws, forecast_dates, standardization_dict, 
                       target_var="HKGDP", train_df=None, n_train_tail=8, interval_width = 0.68):
    """
    Plots a pure BVAR forecast with variable credible intervals.
    Returns summary and full data for downloading.
    """
    
    # 1. Extract the target variable draws (S x h_steps)
    target_draws = forecast_draws[:, :, 0] 
    
    # 2. Un-standardize the draws back to raw values
    mean_scale = standardization_dict[target_var][0]
    std_scale = standardization_dict[target_var][1]
    
    unstd_draws = (target_draws * std_scale) + mean_scale
    
    # 3. Calculate median and dynamic credible bands
    median_forecast = np.median(unstd_draws, axis=0)
    lo_band = np.percentile(unstd_draws, ((100-interval_width)/2), axis=0)
    hi_band = np.percentile(unstd_draws, ((100+interval_width)/2), axis=0)
    
    # ---------------------------------------------------------
    # PLOT THE FORECAST
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    if train_df is not None:
        if isinstance(train_df, pd.DataFrame):
            train_data = train_df[target_var].iloc[-n_train_tail:]
        else:
            train_data = train_df.iloc[-n_train_tail:]
            
        plt.plot(train_data.index, train_data.values, color='black', 
                 linewidth=1.5, label='Historical Data', marker='o', markersize=5)
    
    plt.plot(forecast_dates, median_forecast, 'b-', linewidth=2.5, 
             label='BVAR Median Forecast', marker='o', markersize=5)
    
    for xi, yi in zip(forecast_dates, median_forecast):
        plt.annotate(text=f'{yi: .2f}', xy = (xi, yi), xytext = (0,-25), textcoords = "offset points", ha='center')
    
    if train_df is not None:
        last_hist_x = train_data.index[-1]
        last_hist_y = train_data.values[-1]
        first_fc_x = forecast_dates[0]
        first_fc_y = median_forecast[0]
        plt.plot([last_hist_x, first_fc_x], [last_hist_y, first_fc_y], color='blue', linewidth=2.5)
    
    # DYNAMIC LEGEND: Changes "68%" to whatever the user selected
    plt.fill_between(forecast_dates, lo_band, hi_band, color='blue', alpha=0.2, 
                     label=f'{interval_width}% Credible Interval')
    
    plt.title(f"Pure BVAR Forecast: {target_var}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{target_var}", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    # plt.show() removed for Streamlit compatibility
    
    # ---------------------------------------------------------
    # PREPARE DATA FOR DOWNLOAD
    # ---------------------------------------------------------
    # 1. Summary DataFrame (Rows = dates, Columns = Median, Lower, Upper)
    summary_df = pd.DataFrame({
        'Median': median_forecast,
        f'Lower_{interval_width}': lo_band,
        f'Upper_{interval_width}': hi_band
    }, index=forecast_dates)
    
    # 2. Full Draws DataFrame (Rows = dates, Columns = Draw_1, Draw_2, ...)
    # target_draws is shape (S x h_steps). Transpose it so dates are rows.
    full_df = pd.DataFrame(unstd_draws.T, index=forecast_dates, columns=[f"Draw_{i+1}" for i in range(unstd_draws.shape[0])])
    
    return summary_df, full_df

# %%
def grid_search (df, col_spec, lag_vals, lambda_vals, deltas, decays, training_cutoffs, exog_list, exog_dict, h_steps, n_draws, test_HKGDP, test_dates, cond_setup = None, rolling = False, window = None):
    
    RMSE_list = np.zeros((len(training_cutoffs), len(lag_vals), len(lambda_vals), len(deltas), len(decays)))

    if cond_setup is None:
        condition_dict = None

    for i in range(len(training_cutoffs)):
        cutoff = training_cutoffs[i]
        cutoff_date = pd.to_datetime(cutoff, format="%m/%Y")

        if rolling:
            train_start_date = cutoff_date - pd.DateOffset(months = window*3)
            if train_start_date < df.index[0]:
                print(f"Warning: There is not enough data for training cutoff {i} to have {window} quarters of training data. ")
            train = df[((df.index <= cutoff_date) & (df.index > train_start_date))].dropna()
            print(f"Training data taken from between {train_start_date} and {cutoff_date}")
        else:
            train = df[df.index <= cutoff_date].dropna()

        test_end_date = cutoff_date + pd.DateOffset(months = h_steps*3)
        print(cutoff)
        test  = df[((df.index > cutoff_date) & (df.index <= test_end_date))].dropna()

        ### Robustness to too long forecasting length compared to testing data length

        if len(test) < h_steps:
            new_h_steps = len(test)
            print(f"Test data is too short for cutoff date {cutoff_date}, forecasting length for {cutoff_date} is now {len(test)}.")
        else:
            new_h_steps = h_steps

        ## Constructing a standardized version of qltrain, "Y_stand"

        Y_means = train.mean().values
        Y_stds = train.std().values
        Y_stand = (train - Y_means) / Y_stds

        ## Keep dummies non-standardized

        for exog_var in exog_list:
            Y_stand[exog_var] = train[exog_var]        
        # Y_stand["Covid"] = train["Covid"]
        # Y_stand["GFC"] = train["GFC"]
        # Y_stand["AFC"] = train["AFC"]
        # Y_stand["SARS"] = train["SARS"]

        updated_exog_list = []

        for exog_var in exog_list:
            if np.any(Y_stand[exog_var]):
                updated_exog_list.append(exog_var)

        X_exog = Y_stand[updated_exog_list]
        standardization_dict = {}
        

        for idx, name in enumerate(train.columns):
            standardization_dict[name] = [Y_means[idx], Y_stds[idx]]


        if cond_setup is not(None):
            condition_dict = build_condition_dict(test, cond_setup, standardization_dict)
        for lag in range(len(lag_vals)):
            l = lag_vals[lag]
            for j in range(len(lambda_vals)):
                lambda_val = lambda_vals[j]
                for k in range(len(deltas)):
                    delta = deltas[k]
                    for m in range(len(decays)):
                        decay = decays[m]
                        test_HKGDP = test["HKGDP"]

                        B_draws, Sigma_draws, B_post, S_post = estimate_bvar(Y_stand[col_spec], p=l, lambda_val=lambda_val, 
                                                                            delta=delta, decay=decay, X_exog = X_exog, exog_dict = exog_dict, n_draws = 2000)
                        
            
                        conditional_forecast_draws = dynamic_conditional_forecast(Y_stand[col_spec], p=l, h_steps=new_h_steps, B_draws=B_draws, Sigma_draws=Sigma_draws, condition_dict = condition_dict)
                        
                        RMSE_list[i][lag][j][k][m] = forecast_graph(conditional_forecast_draws, test_HKGDP, p=l, lambda_val = lambda_val, delta = delta, decay = decay, 
                                    standardization_dict=standardization_dict, test_dates=test.index, isplot = False)

    return RMSE_list

# %%
def standardize_df(df):
  


    Y_means = df.mean().values
    Y_stds = df.std().values
    Y_stand = (df - Y_means) / Y_stds

    ## Keep dummies non-standardized
    Y_stand["Covid"] = df["Covid"]
    Y_stand["GFC"] = df["GFC"]


    standardization_dict = {}

    for idx, name in enumerate(df.columns):
        standardization_dict[name] = [Y_means[idx], Y_stds[idx]]

    return Y_stand, standardization_dict

# %%
def clean_timeseries_dataset(df, col_names):
    """
    Cleans a raw macroeconomic dataset by keeping only specified columns 
    and dropping rows with any missing values, maximizing the date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset (likely with many NaNs at the beginning of series).
    col_names : list of str
        The specific columns you want to keep for your analysis.
        
    Returns:
    --------
    pd.DataFrame
        The cleaned dataset containing only the requested columns, with no NaNs,
        sorted chronologically.
    """
    # 1. Validate inputs: Check if the requested columns actually exist
    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Cannot find the following columns in the DataFrame: {missing_cols}")
        
    # 2. Subset the DataFrame to ONLY the columns we care about
    # We use .copy() to avoid Pandas 'SettingWithCopy' warnings later
    df_clean = df[col_names].copy()
    
    # 3. Record how much data we are starting with (for reporting)
    start_rows = len(df_clean)
    
    # 4. Drop any row that has even a single NaN in our target columns
    df_clean = df_clean.dropna()
    
    # 5. Sort chronologically (crucial for time-series BVARs)
    # If the index is datetime, this puts it in perfect order
    try:
        df_clean = df_clean.sort_index()
    except TypeError:
        pass # If index isn't datetime, sorting still works but isn't strictly necessary
        
    # 6. Report the trimming impact
    end_rows = len(df_clean)
    dropped_rows = start_rows - end_rows
    
    print(f"Dataset Cleaning Complete:")
    print(f"  - Starting rows (with NaNs in target cols): {start_rows}")
    print(f"  - Final clean rows (no NaNs): {end_rows}")
    print(f"  - Rows dropped: {dropped_rows} ({(dropped_rows/start_rows)*100:.1f}% of data removed)")
    print(f"  - Start Date: {df_clean.index[0]}")
    print(f"  - End Date:   {df_clean.index[-1]}\n")
    
    return df_clean

# %%
def create_exog_dict(column_names, string_exog_dict):
    """
    Takes as input a dictionary which has variables as keys and the variables 
    they are exogenous to as values, all as strings.
    Outputs the same dictionary but with keys as indices instead of strings.
    Robust to missing keys/values.
    """
    exog_dict = {}

    for col in column_names:
        idx = column_names.index(col)
        
        # Check if the current column is in the dictionary
        if col in string_exog_dict:
            exog_dict[idx] = []
            
            for value in string_exog_dict[col]:
                # ROBUSTNESS CHECK: Only proceed if the value is actually in the selected columns
                if value in column_names:
                    val_idx = column_names.index(value)
                    
                    if val_idx == idx:
                        print("NOTE: THERE IS A VARIABLE WHICH IS EXOGENOUS TO ITSELF.")
                        
                    exog_dict[idx].append(val_idx)
                else:
                    # Optional: Warn the user that a rule was ignored because the variable wasn't selected
                    print(f"NOTE: Exogeneity rule '{col}' -> '{value}' skipped. '{value}' is not in the selected variables.")

    return exog_dict

# %%
def rank_specification(RMSE_list, lag_vals, lambda_vals, deltas, decays, training_cutoffs):
    
    # 1. Flatten the 5D array into a long list of records
    records = []
    for i, cutoff in enumerate(training_cutoffs):
        for lag_idx, l in enumerate(lag_vals):
            for j, lmbda in enumerate(lambda_vals):
                for k, delt in enumerate(deltas):
                    for m, dec in enumerate(decays):
                        records.append({
                            'Cutoff': cutoff,
                            'Lag': l,
                            'Lambda': lmbda,
                            'Delta': delt,
                            'Decay': dec,
                            'RMSE': RMSE_list[i][lag_idx][j][k][m]
                        })

    df_results = pd.DataFrame(records)

    # 2. Calculate the Rank within EACH cutoff date
    # Rank 1 is the best RMSE for that specific cutoff, Rank N is the worst
    df_results['Rank'] = df_results.groupby('Cutoff')['RMSE'].rank(method='min')

    # 3. Calculate the Total Rank Score across all cutoffs
    # Lower is better!
    df_results['Total_Rank'] = df_results.groupby(['Lag', 'Lambda', 'Delta', 'Decay'])['Rank'].transform('sum')

    # 4. Calculate Average RMSE and RMSE Volatility (Standard Deviation across cutoffs)
    agg_stats = df_results.groupby(['Lag', 'Lambda', 'Delta', 'Decay'])['RMSE'].agg(['mean', 'std']).reset_index()
    agg_stats.columns = ['Lag', 'Lambda', 'Delta', 'Decay', 'Mean_RMSE', 'Std_RMSE']

    # 5. Merge ranks back to the aggregated stats
    final_table = agg_stats.merge(
        df_results[['Lag', 'Lambda', 'Delta', 'Decay', 'Total_Rank']].drop_duplicates(), 
        on=['Lag', 'Lambda', 'Delta', 'Decay']
    )

    # 6. Sort to find the winning specification!
    # Primary sort: Total Rank (Robustness). Secondary sort: Mean RMSE (Accuracy).
    final_table = final_table.sort_values(by=['Total_Rank', 'Mean_RMSE'])

    # 7. Print the results beautifully
    print("="*85)
    print("GRID SEARCH RESULTS: RANKED BY ROBUSTNESS ACROSS CUT-OFF DATES")
    print("="*85)
    print(f"{'Lag':<5} {'Lambda':<8} {'Delta':<7} {'Decay':<6} {'T.Rank':<8} {'Mean_RMSE':<12} {'Std_RMSE':<12}")
    print("-"*85)

    # Print top 15 most robust specifications
    for _, row in final_table.head(15).iterrows():
        print(f"{int(row['Lag']):<5} {row['Lambda']:<8.2f} {row['Delta']:<7.2f} {int(row['Decay']):<6} {row['Total_Rank']:<8.0f} {row['Mean_RMSE']:<12.4f} {row['Std_RMSE']:<12.4f}")

    print("-"*85)
    best = final_table.iloc[0]
    print(f"\nWINNING SPECIFICATION:")
    print(f"Lag: {int(best['Lag'])}, Lambda: {best['Lambda']}, Delta: {best['Delta']}, Decay: {int(best['Decay'])}")
    print(f"Average RMSE: {best['Mean_RMSE']:.4f} | RMSE Volatility: {best['Std_RMSE']:.4f}")

# %%
def plot_gdp_with_events(df, event_dates, gdp_col='GDP', line_color='red'):
    """
    Plots a GDP time series from a dataframe and adds colored vertical lines at specified dates.
    
    Parameters:
    - df: pandas DataFrame with a DatetimeIndex/PeriodIndex and a GDP column.
    - event_dates: List of strings representing months/years (e.g., ['2020-03', '2008-09']).
    - gdp_col: The name of the column containing GDP data.
    - line_color: The color of the vertical event lines.
    """
    
    # 1. Ensure the DataFrame index is a datetime type for accurate plotting
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Could not convert DataFrame index to datetime: {e}")
            return

    # 2. Parse the list of string dates into datetime objects
    try:
        parsed_events = pd.to_datetime(event_dates)
    except Exception as e:
        print(f"Error parsing event dates. Ensure they are in a standard format (e.g., 'YYYY-MM'): {e}")
        return

    # 3. Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[gdp_col], label=gdp_col, color='navy', linewidth=2)
    
    # Get the bounds of the data so we don't draw lines way off into the future/past
    x_min, x_max = df.index.min(), df.index.max()

    # 4. Add vertical lines for each event
    for event in parsed_events:
        # Only draw the line if the event date actually falls within the range of your data
        if x_min <= event <= x_max:
            plt.axvline(x=event, color=line_color, linestyle='--', linewidth=1.5, alpha=0.7)
        else:
            print(f"Note: Event date '{event.strftime('%Y-%m')}' is outside the DataFrame's date range. Skipping.")

    # 5. Formatting
    plt.title(f'{gdp_col} Over Time with Training Cutoff Markers', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(gdp_col, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Automatically rotates and formats the x-axis dates so they don't overlap
    plt.gcf().autofmt_xdate() 
    
    plt.legend()
    plt.tight_layout()
    #plt.show()
# %%
def process_data(df, new_cols, QoQ = True):

    df.columns = ["Quarter", "HKGDP", "HKGDP_yoy", "Imports", "Exports", "RSV", "HSI", "PPI", "PST_Volume", "FFR", "China_PMI_NEO", "CCPI"]

    cols = ["HKGDP", "HKGDP_yoy", "Imports", "Exports", "RSV", "HSI", "PPI", "PST_Volume", "FFR", "China_PMI_NEO", "CCPI"]
    
    df=df.drop(index=0)

    ## Changing data to correct types

    df['Quarter'] = pd.to_datetime(df['Quarter'], format="%m/%Y", errors="coerce")
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.set_index('Quarter')

    ## ADJUSTING IMPORTS, EXPORTS, and RETAIL SALES VALUE FOR INFLATION AND MAKING THEM YOY%

    df["Imports_adj"] = df["Imports"] / df["CCPI"]
    df["Exports_adj"] = df["Exports"] / df["CCPI"]
    df["RSV_adj"] = df["RSV"] / df["CCPI"]

    ## Taking YOY% growth for each (INCLUDING China PMI_NEO)

    df["Imports"] = (df["Imports_adj"] - df["Imports_adj"].shift(4))/df["Imports_adj"].shift(4)
    df["Exports"] = (df["Exports_adj"] - df["Exports_adj"].shift(4))/df["Exports_adj"].shift(4)
    df["RSV"] = (df["RSV_adj"] - df["RSV_adj"].shift(4))/df["RSV_adj"].shift(4)
    df["China_PMI_NEO"] = (df["China_PMI_NEO"] - df["China_PMI_NEO"].shift(4))/df["China_PMI_NEO"].shift(4)

    df=df.drop(columns = ["Imports_adj", "Exports_adj", "RSV_adj"])

    ## Annualizing Quarterly GDP

    df["HKGDP"] = ((1+(df["HKGDP"])/100)**4 - 1)

    ## Taking log difference of Hang Seng Index

    df["HSI_log"] = np.log(df["HSI"])
    df["HSI"] = df["HSI_log"].diff()

    ## Taking first difference of Federal Funds Rate

    df["FFR"] = df["FFR"].diff()

    ## Dividing select variables by 100 to get relatively consistent orders of magnitude

    df["PPI"] = df["PPI"]/100
    df["PST_Volume"] = df["PST_Volume"]/100


    df = df.drop(columns = ["HSI_log", "CCPI"])

    # Get GDP in comparable units as before

    df = df*100

    if QoQ:
        ### Option to use HKGDP QoQ

        df=df.drop(columns = "HKGDP_yoy")
    else:
        ### Option to use HKGDP YoY

        df["HKGDP"] = df["HKGDP_yoy"]/100
        df=df.drop(columns = "HKGDP_yoy")

    df = clean_timeseries_dataset(df, new_cols)

    #==============================================#
    ## ADDING A COVID DUMMY FOR COMPARISON
    #==============================================#

    # Define COVID period (adjust as needed)

    covid_start = '2020Q1'
    covid_end = '2021Q2'

    df["Covid"] = ((df.index < covid_end) & (df.index > covid_start)) 
    df['Covid'] = df['Covid'].astype(int)

    #==============================================#
    ## ADDING A GLOBAL FINANCIAL CRISIS DUMMY
    #==============================================#

    # Define GFC period (adjust as needed)

    gfc_start = '2008Q3'
    gfc_end = '2009Q2'

    df["GFC"] = ((df.index < gfc_end) & (df.index > gfc_start)) 
    df['GFC'] = df['GFC'].astype(int)

    return df
# %%
def slice_df(df, cutoff_date):
    ## INITIALIZING TRAINING DATA BASED ON 2023Q3 CUTOFF


    train = df.loc[df.index <= cutoff_date].dropna()
    test  = df.loc[df.index > cutoff_date].dropna()


    ## Constructing a standardized version of qltrain, "Y_stand"

    Y_means = train.mean().values
    Y_stds = train.std().values
    Y_stand = (train - Y_means) / Y_stds

    ## Keep dummies non-standardized
    Y_stand["Covid"] = train["Covid"]
    Y_stand["GFC"] = train["GFC"]


    standardization_dict = {}

    for idx, name in enumerate(train.columns):
        standardization_dict[name] = [Y_means[idx], Y_stds[idx]]

    return Y_stand, train, test, standardization_dict
# %%
def fit_arima_and_eval(p: int, d: int, q: int, train: pd.Series, test: pd.Series, isplot = True):
    """
    Fit ARIMA(p,d,q) on train series and evaluate on test series.
    Returns a dict with model order, coefficients (as string), AIC, BIC, RMSE.
    """
    result = {
        "p": p, "d": d, "q": q,
        "coefficients": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "rmse": np.nan,
        "status": "failed"
    }

    # Ensure numeric numpy arrays
    train = pd.to_numeric(train, errors="coerce").dropna()
    test = pd.to_numeric(test, errors="coerce").dropna()

    if len(train) < (p + d + q + 1):
        result["status"] = "insufficient_train"
        return result

    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        # coefficients as "name:val; ..." single-line string
        coeffs = "; ".join([f"{k}:{v:.6g}" for k, v in model_fit.params.items()])

        # Forecast same length as test
        n_steps = len(test)
        if n_steps == 0:
            forecast = np.array([])
        else:
            forecast = model_fit.forecast(steps=n_steps)

        # Convert to numeric arrays for RMSE
        y_true = np.asarray(test, dtype=float)
        y_pred = np.asarray(forecast, dtype=float)

        rmse = np.nan
        if len(y_true) == len(y_pred) and len(y_true) > 0:
            rmse = sqrt(mean_squared_error(y_true, y_pred))

        result.update({
            "coefficients": coeffs,
            "aic": float(model_fit.aic),
            "bic": float(model_fit.bic),
            "rmse": float(rmse) if not np.isnan(rmse) else np.nan,
            "status": "ok"
        })
    except Exception as e:
        # capture failure reason in status for debugging
        result["status"] = f"error: {str(e)}"

    if not(isplot):
        return rmse

    ## Forecasting plot
    plt.figure(figsize=(10,5))
    plt.plot(train["HKGDP"].index, train["HKGDP"], label='Train', color='C0')
    plt.plot(test["HKGDP"].index, test["HKGDP"], label='Test', color='C1')
    plt.plot(forecast.index, forecast, label='Forecast', color='C2', linestyle='--')

    plt.title(f"ARIMA({p},{d},{q}) Forecast — RMSE {rmse:.4g}")
    plt.xlabel("Quarter")
    plt.ylabel("HKGDP")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    return result


# %%
def arma_grid_search (Y, p_vals, q_vals, training_cutoffs, h_steps):

    ARMA_RMSE = np.zeros((len(training_cutoffs), len(p_vals), len(q_vals)))


    for i in range(len(training_cutoffs)):

        cutoff = training_cutoffs[i]
        cutoff_date = pd.to_datetime(cutoff, format="%m/%Y")
        train = df.loc[df.index <= cutoff_date].dropna()
        test_end_date = cutoff_date + pd.DateOffset(months = 30)


        test  = df.loc[((df.index > cutoff_date) & (df.index <= test_end_date))].dropna()

        for j in range(len(p_vals)):
            p = p_vals[j]
            for k in range(len(q_vals)):
                q = q_vals[k]

                rmse = fit_arima_and_eval(p, 0, q, train["HKGDP"],test["HKGDP"], isplot=False)
                ARMA_RMSE[i][j][k] = rmse

    return ARMA_RMSE


# %%
def rank_arma_specification(RMSE_list, p_vals, q_vals, training_cutoffs):
    """
    Ranks ARMA(p,q) specifications based on robustness across different time cutoffs.
    
    Parameters:
    -----------
    RMSE_list : 3D numpy array (len(training_cutoffs) x len(p_vals) x len(q_vals))
    p_vals : list of integers (e.g., [1, 2, 3, 4])
    q_vals : list of integers (e.g., [0, 1, 2])
    training_cutoffs : list of strings (e.g., ["09/2023", "12/2019", ...])
    """
    
    # 1. Flatten the 3D array into a long list of records
    records = []
    for i, cutoff in enumerate(training_cutoffs):
        for j, p in enumerate(p_vals):
            for k, q in enumerate(q_vals):
                records.append({
                    'Cutoff': cutoff,
                    'AR_p': p,
                    'MA_q': q,
                    'RMSE': RMSE_list[i][j][k]
                })

    df_results = pd.DataFrame(records)

    # 2. Calculate the Rank within EACH cutoff date
    # Rank 1 is the best RMSE for that specific cutoff, Rank N is the worst
    df_results['Rank'] = df_results.groupby('Cutoff')['RMSE'].rank(method='min')

    # 3. Calculate the Total Rank Score across all cutoffs
    # Lower is better!
    df_results['Total_Rank'] = df_results.groupby(['AR_p', 'MA_q'])['Rank'].transform('sum')

    # 4. Calculate Average RMSE and RMSE Volatility (Standard Deviation across cutoffs)
    agg_stats = df_results.groupby(['AR_p', 'MA_q'])['RMSE'].agg(['mean', 'std']).reset_index()
    agg_stats.columns = ['AR_p', 'MA_q', 'Mean_RMSE', 'Std_RMSE']

    # 5. Merge ranks back to the aggregated stats
    final_table = agg_stats.merge(
        df_results[['AR_p', 'MA_q', 'Total_Rank']].drop_duplicates(), 
        on=['AR_p', 'MA_q']
    )

    # 6. Sort to find the winning specification!
    # Primary sort: Total Rank (Robustness). Secondary sort: Mean RMSE (Accuracy).
    final_table = final_table.sort_values(by=['Total_Rank', 'Mean_RMSE'])

    # 7. Print the results beautifully
    print("="*70)
    print("ARMA GRID SEARCH: RANKED BY ROBUSTNESS ACROSS CUT-OFF DATES")
    print("="*70)
    print(f"{'AR(p)':<7} {'MA(q)':<7} {'T.Rank':<8} {'Mean_RMSE':<12} {'Std_RMSE':<12}")
    print("-"*70)

    # Print top 15 most robust specifications
    for _, row in final_table.head(15).iterrows():
        print(f"{int(row['AR_p']):<7} {int(row['MA_q']):<7} {row['Total_Rank']:<8.0f} {row['Mean_RMSE']:<12.4f} {row['Std_RMSE']:<12.4f}")

    print("-"*70)
    best = final_table.iloc[0]
    print(f"\nWINNING ARMA SPECIFICATION:")
    print(f"AR({int(best['AR_p'])},{int(best['MA_q'])})")
    print(f"Average RMSE: {best['Mean_RMSE']:.4f} | RMSE Volatility: {best['Std_RMSE']:.4f}")
    
    return final_table

# %%
def plot_gdp_with_events(df, event_dates, gdp_col='GDP', line_color='red'):
    """
    Plots a GDP time series from a dataframe and adds colored vertical lines at specified dates.
    
    Parameters:
    - df: pandas DataFrame with a DatetimeIndex/PeriodIndex and a GDP column.
    - event_dates: List of strings representing months/years (e.g., ['2020-03', '2008-09']).
    - gdp_col: The name of the column containing GDP data.
    - line_color: The color of the vertical event lines.
    """
    
    # 1. Ensure the DataFrame index is a datetime type for accurate plotting
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Could not convert DataFrame index to datetime: {e}")
            return

    # 2. Parse the list of string dates into datetime objects
    try:
        parsed_events = pd.to_datetime(event_dates)
    except Exception as e:
        print(f"Error parsing event dates. Ensure they are in a standard format (e.g., 'YYYY-MM'): {e}")
        return

    # 3. Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[gdp_col], label=gdp_col, color='navy', linewidth=2)
    
    # Get the bounds of the data so we don't draw lines way off into the future/past
    x_min, x_max = df.index.min(), df.index.max()

    # 4. Add vertical lines for each event
    for event in parsed_events:
        # Only draw the line if the event date actually falls within the range of your data
        if x_min <= event <= x_max:
            plt.axvline(x=event, color=line_color, linestyle='--', linewidth=1.5, alpha=0.7)
        else:
            print(f"Note: Event date '{event.strftime('%Y-%m')}' is outside the DataFrame's date range. Skipping.")

    # 5. Formatting
    plt.title(f'{gdp_col} Over Time with Training Cutoff Markers', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(gdp_col, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Automatically rotates and formats the x-axis dates so they don't overlap
    plt.gcf().autofmt_xdate() 
    
    plt.legend()
    plt.tight_layout()
    #plt.show()

def VAR_grid_search(df, col_spec, training_cutoffs, lag_vals, h_steps, target_var=None):
    """
    Grid searches unrestricted VAR lag lengths and returns an array of RMSEs.
    """
    # Initialize with NaNs instead of zeros. 
    # If a model fails to fit, it will remain NaN rather than falsely reporting 0 error.
    RMSE_list = np.full([len(training_cutoffs), len(lag_vals)], np.nan)

    # Default to the first column in col_spec if no target is specified
    if target_var is None:
        target_var = col_spec[0]

    # Use different index variables for the outer and inner loops!
    for c_idx, cutoff in enumerate(training_cutoffs):
        
        # Parse cutoff date (added a fallback in case your strings aren't strictly %m/%Y)
        try:
            cutoff_date = pd.to_datetime(cutoff, format="%m/%Y")
        except:
            cutoff_date = pd.to_datetime(cutoff)
            
        # Split Data
        train = df[df.index <= cutoff_date][col_spec].dropna()
        test_end_date = cutoff_date + pd.DateOffset(months=h_steps*3) # Assuming quarterly data
        test = df[((df.index > cutoff_date) & (df.index <= test_end_date))][col_spec].dropna()

        # Handle test set length
        if len(test) < h_steps:
            new_h_steps = len(test)
            print(f"Test data too short for cutoff {cutoff_date}, using {new_h_steps} steps.")
        else:
            new_h_steps = h_steps
            
        if len(test) == 0:
            print(f"No test data available for {cutoff_date}. Skipping.")
            continue

        # Inner loop over lags
        for l_idx, lag_val in enumerate(lag_vals):  
            
            # Check if we even have enough data to estimate this lag
            if len(train) <= lag_val:
                print(f"Not enough training data for lag {lag_val} at {cutoff_date}. Skipping.")
                continue
                
            try:
                # Fit the model
                model = VAR(train)
                fitted_model = model.fit(lag_val)
                
                # Generate Forecast
                # statsmodels requires the last 'lag_val' observations to make the forecast
                forecast_input = train.values[-lag_val:]
                forecast = fitted_model.forecast(y=forecast_input, steps=new_h_steps)
                
                # Align forecast with actual test data
                # forecast is a numpy array, let's make it a dataframe for easy slicing
                forecast_df = pd.DataFrame(forecast, 
                                           index=test.index[:new_h_steps], 
                                           columns=col_spec)
                
                # Calculate RMSE for the specific target variable
                actuals = test[target_var].values[:new_h_steps]
                preds = forecast_df[target_var].values
                
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                
                # Store in the array
                RMSE_list[c_idx, l_idx] = rmse
                
            except Exception as e:
                # Unrestricted VARs frequently fail with "Singular Matrix" errors on high lags.
                # This prevents the whole script from crashing.
                # print(f"Failed to fit lag {lag_val} at {cutoff_date}: {str(e)[:50]}...")
                pass 

    return RMSE_list

# %%
def rank_var_specification(RMSE_list, lag_vals, training_cutoffs):
    """
    Ranks VAR lag specifications based on robustness across different time cutoffs.
    
    Parameters:
    -----------
    RMSE_list : 2D numpy array (len(training_cutoffs) x len(lag_vals))
    lag_vals : list of integers (e.g., [1, 2, 3, 4])
    training_cutoffs : list of strings (e.g., ["09/2023", "12/2019", ...])
    """
    
    # 1. Flatten the 2D array into a long list of records
    records = []
    for i, cutoff in enumerate(training_cutoffs):
        for j, lag in enumerate(lag_vals):
            records.append({
                'Cutoff': cutoff,
                'Lag': lag,
                'RMSE': RMSE_list[i][j]
            })

    df_results = pd.DataFrame(records)

    # 2. Calculate the Rank within EACH cutoff date
    # Rank 1 is the best RMSE for that specific cutoff, Rank N is the worst.
    # Note: NaN values (from failed unrestricted VARs) automatically get the worst rank.
    df_results['Rank'] = df_results.groupby('Cutoff')['RMSE'].rank(method='min')

    # 3. Calculate the Total Rank Score across all cutoffs
    # Lower is better!
    df_results['Total_Rank'] = df_results.groupby('Lag')['Rank'].transform('sum')

    # 4. Calculate Average RMSE and RMSE Volatility (Standard Deviation across cutoffs)
    # We use nanmean/nanstd implicitly by telling pandas to skip NaNs during aggregation
    agg_stats = df_results.groupby('Lag')['RMSE'].agg(['mean', 'std']).reset_index()
    agg_stats.columns = ['Lag', 'Mean_RMSE', 'Std_RMSE']

    # 5. Merge ranks back to the aggregated stats
    final_table = agg_stats.merge(
        df_results[['Lag', 'Total_Rank']].drop_duplicates(), 
        on='Lag'
    )

    # 6. Sort to find the winning specification!
    # Primary sort: Total Rank (Robustness). Secondary sort: Mean RMSE (Accuracy).
    final_table = final_table.sort_values(by=['Total_Rank', 'Mean_RMSE'])

    # 7. Print the results beautifully
    print("="*60)
    print("VAR GRID SEARCH: RANKED BY ROBUSTNESS ACROSS CUT-OFF DATES")
    print("="*60)
    print(f"{'Lag(p)':<8} {'T.Rank':<8} {'Mean_RMSE':<12} {'Std_RMSE':<12}")
    print("-"*60)

    # Print all lag specifications
    for _, row in final_table.iterrows():
        lag_str = str(int(row['Lag']))
        
        # Format RMSE strings, handling cases where the model failed entirely
        mean_rmse_str = f"{row['Mean_RMSE']:.4f}" if pd.notna(row['Mean_RMSE']) else "Failed"
        std_rmse_str = f"{row['Std_RMSE']:.4f}" if pd.notna(row['Std_RMSE']) else "Failed"
        
        print(f"{lag_str:<8} {row['Total_Rank']:<8.0f} {mean_rmse_str:<12} {std_rmse_str:<12}")

    print("-"*60)
    
    # Check if the absolute best model actually successfully estimated
    if pd.notna(final_table.iloc[0]['Mean_RMSE']):
        best = final_table.iloc[0]
        print(f"\nWINNING VAR SPECIFICATION:")
        print(f"VAR({int(best['Lag'])})")
        print(f"Average RMSE: {best['Mean_RMSE']:.4f} | RMSE Volatility: {best['Std_RMSE']:.4f}")
    else:
        print("\nWARNING: All models failed to estimate across all cutoffs.")
    
    return final_table



