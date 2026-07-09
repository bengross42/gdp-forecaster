import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the functions you saved in Step 2
from bvar_model import (
    process_data,
    standardize_df,
    slice_df,
    construct_var_matrix,
    minnesota_prior,
    estimate_bvar, 
    unconditional_forecast,
    dynamic_conditional_forecast,
    plot_pure_forecast,
    VAR_grid_search, 
    rank_var_specification,
    create_exog_dict,
    plot_gdp_with_events 
)

# ==========================================
# PRO-TIP: CACHING
# ==========================================
@st.cache_data
def load_and_process_data(uploaded_file, new_cols, QoQ):
    df_raw = pd.read_excel(uploaded_file)
    df = process_data(df_raw, new_cols, QoQ)
    return df
# ==========================================

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GDP Forecasting App", layout="wide")
st.title("📉 Quarterly GDP Forecasting Engine")
st.markdown("Upload your Excel file, configure the model, and generate forecasts.")

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header("1. Data Upload & Selection")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx'])

# Only show the rest of the sidebar if a file is uploaded
if uploaded_file is not None:

    available_variables = ["Imports", "Exports", "RSV", "HSI", "PPI", "PST_Volume", "FFR", "China_PMI_NEO"]
    
    selected_GDP = st.sidebar.multiselect(
        label="Choose HKGDP specification for the BVAR",
        options=["HKGDP_qoq", "HKGDP_yoy"],
        key = "sel_gdp"
    )

    selected_variables = st.sidebar.multiselect(
        label="Choose OTHER variables for the BVAR",
        options=available_variables,
        key = "sel_vars"
    )

    if len(selected_GDP) > 0 and selected_GDP[0] == "HKGDP_qoq":
        QoQ = True
    else:
        QoQ = False

    if len(selected_variables) == 0 and len(selected_GDP) == 0:
        st.warning("Please select at least one variable from the sidebar.")

    else:
        try:
            final_cols = ["HKGDP"] + selected_variables
            df = load_and_process_data(uploaded_file, final_cols, QoQ)
            
            st.success("Data processed successfully!")
            st.dataframe(df) 
            
           # ==========================================
            # RECOMMENDED SPECIFICATIONS
            # ==========================================
            
            # 1. DEFINE THE CALLBACK FUNCTIONS FIRST
            # These run behind the scenes BEFORE the page redraws
            def load_rec_1():
                st.session_state.sel_gdp = ["HKGDP_qoq"]
                st.session_state.sel_vars = ["Imports", "RSV", "FFR"]
                st.session_state.lag_val = 2
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.2
                st.session_state.decay_val = 1

            def load_rec_2():
                st.session_state.sel_gdp = ["HKGDP_qoq"]
                st.session_state.sel_vars = available_variables # Selects ALL
                st.session_state.lag_val = 4
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.3
                st.session_state.decay_val = 1

            def load_rec_3():
                st.session_state.sel_gdp = ["HKGDP_yoy"]
                st.session_state.sel_vars = ["Imports", "RSV", "FFR"]
                st.session_state.lag_val = 6
                st.session_state.lambda_val = 0.4
                st.session_state.delta_val = 0.2
                st.session_state.decay_val = 1

            def load_rec_4():
                st.session_state.sel_gdp = ["HKGDP_yoy"]
                st.session_state.sel_vars = available_variables # Selects ALL
                st.session_state.lag_val = 2
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.5
                st.session_state.decay_val = 1

            # 2. ASSIGN CALLBACKS TO BUTTONS
            st.subheader("🎯 Recommended Specifications")
            st.markdown("Click a button below to instantly load the parameters into the sidebar.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. QoQ Baseline (Parsimonious)**")
                # Notice we use on_click= instead of if st.button
                st.button("Load: QoQ | Core Vars | p=2, λ=.25, δ=.2", on_click=load_rec_1, use_container_width=True)
                    
                st.markdown("**3. YoY Baseline (Parsimonious)**")
                st.button("Load: YoY | Core Vars | p=6, λ=.4, δ=.2", on_click=load_rec_3, use_container_width=True)

            with col2:
                st.markdown("**2. QoQ Full Model**")
                st.button("Load: QoQ | All Vars | p=4, λ=.25, δ=.3", on_click=load_rec_2, use_container_width=True)
                    
                st.markdown("**4. YoY Full Model**")
                st.button("Load: YoY | All Vars | p=2, λ=.25, δ=.5", on_click=load_rec_4, use_container_width=True)
            
            st.divider() 
            
            col_spec = df.columns.tolist()
            
            # ==========================================
            # MOVE DISPLAY OPTIONS & PARAMETERS HERE
            # ==========================================
            st.sidebar.header("2. Model Parameters")
            lag_val = st.sidebar.slider("Lag Length (p)", min_value=1, max_value=8, value=4, key = "lag_val")
            lambda_val = st.sidebar.slider("Prior Tightness (Lambda)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, key = "lambda_val")
            delta = st.sidebar.slider("Cross Variable Tightness (Delta)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, key = "delta_val")
            decay = st.sidebar.slider("Shrinkage Over Time (Decay)", min_value=1.0, max_value=4.0, value=2.0, step=1.0, key = "decay_val")
            n_draws = st.sidebar.slider("Number Draws", min_value=500, max_value=8000, value=2000, step=500)
            h_steps = st.sidebar.slider("Forecast Horizon (h_steps)", min_value=1, max_value=8, value=4, step=1)
            
            # Ensure integers
            n_draws = int(n_draws)
            h_steps = int(h_steps)

            st.sidebar.header("3. Display Options")
            show_plot = st.sidebar.checkbox("Show Coefficient Plot")
            show_equation = st.sidebar.checkbox("Show Estimating Equation")
            target_var_idx = st.sidebar.number_input("Variable Index to Plot/Print (0 = first column)", value=0)

            # ==========================================
            # THE FORECAST BUTTON
            # ==========================================
            if st.button("🚀 Click Here to Run Forecast", type="primary", use_container_width=True):
                
                with st.spinner(f"Estimating BVAR ({n_draws} draws) and forecasting {h_steps} steps ahead..."):
                    
                    # --- DATA PREP ---
                    Y_stand, standardization_dict = standardize_df(df)
                    diff_exog_dict = {
                        "HKGDP": ["FFR", "China_PMI_NEO"], "Imports": ["FFR", "China_PMI_NEO"], 
                        "HSI": ["FFR", "China_PMI_NEO"], "PPI": ["FFR", "China_PMI_NEO"], 
                        "Exports": ["FFR", "China_PMI_NEO"], "RSV": ["FFR", "China_PMI_NEO"], 
                        "PST_Volume": ["FFR", "China_PMI_NEO"], "FFR": ["China_PMI_NEO"], 
                        "China_PMI_NEO": ["FFR"]
                    }
                    exog_dict = create_exog_dict(final_cols, diff_exog_dict)
                    exog_list = ["Covid", "GFC"]

                    X_exog = Y_stand[exog_list]
                    future_exog = None

                    # --- DYNAMICALLY SET PLOT/EQUATION ARGS ---
                    # If checkbox is true, pass the index. If false, pass None (which turns it off in your function)
                    plot_idx = target_var_idx if show_plot else None
                    eq_idx = target_var_idx if show_equation else None

                    # --- ESTIMATE BVAR ---
                    B_draws, Sigma_draws, B_post, S_post = estimate_bvar(
                        Y_stand[final_cols], 
                        p=lag_val, 
                        lambda_val=lambda_val, 
                        delta=delta, 
                        decay=decay, 
                        X_exog=X_exog, 
                        exog_dict=exog_dict, 
                        n_draws=n_draws, # Changed from 8000 to your slider variable
                        plot_var_idx=plot_idx, 
                        print_eq_idx=eq_idx, 
                        var_names=final_cols, 
                        exog_names=exog_list
                    )

                    # --- DISPLAY ESTIMATING EQUATION ---
                    # (Note: Your function uses print(), which Streamlit captures in a grey box at the bottom)
                    if show_equation:
                        st.subheader("📝 Estimating Equation")
                        st.caption("*See captured output below*")

                    # --- DISPLAY COEFFICIENT PLOT ---
                    if show_plot:
                        st.subheader("📊 Posterior Coefficients")
                        # Grab the figure created inside estimate_bvar and display it
                        st.pyplot(plt.gcf())
                        plt.clf() # Clear it so it doesn't bleed into the forecast plot

                    # --- GENERATE FORECAST ---
                    conditional_forecast_draws = dynamic_conditional_forecast(
                        Y_stand[final_cols], lag_val, B_draws, Sigma_draws, h_steps, 
                        future_exog=future_exog, condition_dict=None
                    )

                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=h_steps+1, freq='QS-MAR')[1:]

                    # --- DISPLAY FORECAST PLOT ---
                    st.subheader("📈 GDP Forecast")
                    plot_pure_forecast(
                        conditional_forecast_draws, future_dates, standardization_dict, 
                        target_var="HKGDP", train_df=df, n_train_tail=16
                    )
                    st.pyplot(plt.gcf())
                    plt.clf() 

                st.success("Forecast Generated Successfully!")

        except Exception as e:
            st.error(f"Failed to process file or run model. Error: {e}")
