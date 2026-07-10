import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import contextlib

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
    integrated_forecast_graph,
    VAR_grid_search, 
    rank_var_specification,
    create_exog_dict,
    plot_gdp_with_events 
)

# ==========================================
# PRO-TIP: CACHING
# ==========================================

@st.cache_data
def load_covid_control_data():
    """
    Loads the static COVID control variable from the app folder.
    """
    # If you saved as CSV:
    covid_df = pd.read_csv("covid_control_data.csv")
    
    return covid_df

@st.cache_data
def load_and_process_data(uploaded_file, new_cols, covid_df, QoQ):
    df_raw = pd.read_excel(uploaded_file)
    df = process_data(df_raw, new_cols, covid_df, QoQ)
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
    
    selected_GDP = st.sidebar.selectbox(
        label="Choose HKGDP specification for the BVAR",
        options=["HKGDP_qoq", "HKGDP_yoy"],
        key = "sel_gdp"
    )

    selected_variables = st.sidebar.multiselect(
        label="Choose OTHER variables for the BVAR",
        options=available_variables,
        key = "sel_vars"
    )

    # --- NEW: EXOGENOUS CONTROL VARIABLES ---
    # Replace the names below with the EXACT column names from your merged dataframe
    available_exog_controls = ["Covid_dummy", "GFC_dummy", "Protest_dummy", "HK_Covid_Proxy", "China_Covid_Proxy"]
    
    selected_exog = st.sidebar.multiselect(
        label="Choose Exogenous Control Variables",
        options=available_exog_controls,
        key="sel_exog"
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

            covid_df = load_covid_control_data()
            df = load_and_process_data(uploaded_file, final_cols, covid_df, QoQ)

            # for exog_var in available_exog_controls:
            #     if exog_var not in selected_exog:
            #         df=df.drop(columns = exog_var)
            
            st.success("Data processed successfully!")
            st.dataframe(df) 
            
           # ==========================================
            # RECOMMENDED SPECIFICATIONS
            # ==========================================
            
            # 1. DEFINE THE CALLBACK FUNCTIONS FIRST
            # These run behind the scenes BEFORE the page redraws

            default_exog = [x for x in available_exog_controls if x not in ["Covid_dummy", "Protest_dummy"]]

            def load_rec_1():
                st.session_state.sel_gdp = ["HKGDP_qoq"]
                st.session_state.sel_vars = ["Imports", "RSV", "FFR"]
                st.session_state.lag_val = 2
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.2
                st.session_state.decay_val = 1
                st.session_state.sel_exog = default_exog

            def load_rec_2():
                st.session_state.sel_gdp = ["HKGDP_qoq"]
                st.session_state.sel_vars = available_variables # Selects ALL
                st.session_state.lag_val = 4
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.3
                st.session_state.decay_val = 1
                st.session_state.sel_exog = default_exog

            def load_rec_3():
                st.session_state.sel_gdp = ["HKGDP_yoy"]
                st.session_state.sel_vars = ["Imports", "RSV", "FFR"]
                st.session_state.lag_val = 6
                st.session_state.lambda_val = 0.4
                st.session_state.delta_val = 0.2
                st.session_state.decay_val = 1
                st.session_state.sel_exog = default_exog

            def load_rec_4():
                st.session_state.sel_gdp = ["HKGDP_yoy"]
                st.session_state.sel_vars = available_variables # Selects ALL
                st.session_state.lag_val = 2
                st.session_state.lambda_val = 0.25
                st.session_state.delta_val = 0.5
                st.session_state.decay_val = 1
                st.session_state.sel_exog = default_exog

            # 2. ASSIGN CALLBACKS TO BUTTONS
            st.subheader("🎯 Recommended Specifications Based on Previous Hyperparameter Optimization")
            st.markdown("Click a button below to instantly load the parameters into the sidebar.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. YoY Baseline (Parsimonious)**")
                st.button("Load: YoY | Core Vars | p=6, λ=.4, δ=.2", on_click=load_rec_3, use_container_width=True)
                    
                st.markdown("**2. QoQ Baseline (Parsimonious)**")
                st.button("Load: QoQ | Core Vars | p=2, λ=.25, δ=.2", on_click=load_rec_1, use_container_width=True)

            with col2:
                st.markdown("**3. YoY Full Model**")
                st.button("Load: YoY | All Vars | p=2, λ=.25, δ=.5", on_click=load_rec_4, use_container_width=True)
                    
                st.markdown("**4. QoQ Full Model**")
                st.button("Load: QoQ | All Vars | p=4, λ=.25, δ=.3", on_click=load_rec_2, use_container_width=True)
            
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
            interval_width = st.sidebar.slider("Confidence Level", min_value=1, max_value=99, value=68, step=1)
            
            # Ensure integers
            n_draws = int(n_draws)
            h_steps = int(h_steps)

            st.sidebar.header("3. Display Options")
            show_plot = st.sidebar.checkbox("Show Coefficient Plot")
            show_equation = st.sidebar.checkbox("Show Estimating Equation")
            target_var_idx = st.sidebar.number_input("Variable Index to Plot/Print (0 = first column)", value=0)

                        # ==========================================
            # NEW: SCENARIO ANALYSIS TOGGLE
            # ==========================================
            st.sidebar.subheader("4. Scenario Analysis")
            run_scenario = st.sidebar.checkbox("Condition Future Variables", key="scenario_toggle")
            
            condition_dict = None
    
            if run_scenario:
                st.sidebar.error("⚠️ WARNING: Input values must be in STANDARDIZED units (e.g., 1.0 for a 1 std dev shock), NOT raw percentages!")
                
                # --- DYNAMIC FONT SIZE LOGIC ---
                # Start at 16px, shrink by ~1.2px for every step over 1, minimum 9px
                dynamic_font_size = max(9, 16 - (h_steps - 1) * 1.2)
                
                # Inject CSS directly into the Streamlit sidebar
                custom_css = f"""
                <style>
                /* Target text inputs specifically inside the sidebar */
                section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
                    font-size: {dynamic_font_size}px !important;
                    padding: 4px 6px !important; /* Shrink the box padding to fit more horizontally */
                }}
                </style>
                """
                st.markdown(custom_css, unsafe_allow_html=True)
                # ----------------------------------

                condition_dict = {}
                
                # Iterate through selected variables (which excludes HKGDP)
                for i, var_name in enumerate(selected_variables):
                    idx_in_final = final_cols.index(var_name) 
                    
                    st.sidebar.markdown(f"**{var_name}**")
                    
                    # Create horizontal columns so inputs fit in the sidebar
                    input_cols = st.sidebar.columns(h_steps)
                    vals = []
                    
                    for h in range(h_steps):
                        with input_cols[h]:
                            val = st.text_input(
                                label=f"t+{h+1}", 
                                key=f"cond_{var_name}_{h}", 
                                label_visibility="collapsed"
                            )
                            
                            if val.strip() != "":
                                try:
                                    vals.append(float(val))
                                except ValueError:
                                    st.sidebar.error(f"Invalid number for {var_name} t+{h+1}")
                                    
                    if len(vals) > 0:
                        condition_dict[idx_in_final] = vals
            
            st.sidebar.subheader("5. Download Options")
            dl_summary = st.sidebar.checkbox("Download Summary Forecast (Median & Bands)")
            dl_full = st.sidebar.checkbox("Download Full Forecast Draws (All MCMC Samples)")

            # ===========================================================================================================================================
            # THE FORECAST BUTTON
            # ===========================================================================================================================================
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

                    # --- DYNAMIC EXOG HANDLING ---
                    exog_list = selected_exog

                    if len(exog_list) > 0:
                        X_exog = Y_stand[exog_list]
                    else:
                        X_exog = None # Safely passes None if user unchecks all boxes
                        
                    future_exog = None


                    # --- DYNAMICALLY SET PLOT/EQUATION ARGS ---
                    plot_idx = target_var_idx if show_plot else None
                    eq_idx = target_var_idx if show_equation else None

                    # --- ESTIMATE BVAR ---
                    # Create a temporary string buffer to catch the print() statement
                    equation_buffer = io.StringIO()
                    
                    # Run the function, but redirect any print() outputs to our buffer
                    with contextlib.redirect_stdout(equation_buffer):
                        B_draws, Sigma_draws, B_post, S_post = estimate_bvar(
                            Y_stand[final_cols], 
                            p=lag_val, 
                            lambda_val=lambda_val, 
                            delta=delta, 
                            decay=decay, 
                            X_exog=X_exog, 
                            exog_dict=exog_dict, 
                            n_draws=n_draws, 
                            plot_var_idx=plot_idx, 
                            print_eq_idx=eq_idx, 
                            var_names=final_cols, 
                            exog_names=exog_list
                        )

                    # --- DISPLAY ESTIMATING EQUATION ---
                    if show_equation:
                        # Extract the text from the buffer
                        equation_text = equation_buffer.getvalue()
                        
                        if equation_text.strip(): # Check if it's not empty
                            st.subheader("📝 Estimating Equation")
                            # st.code() renders it in a nice monospace font, preserving your spacing!
                            st.code(equation_text, language=None)

                    # --- DISPLAY COEFFICIENT PLOT ---
                    if show_plot:
                        st.subheader("📊 Posterior Coefficients")
                        # Grab the figure created inside estimate_bvar and display it
                        st.pyplot(plt.gcf())
                        plt.clf() # Clear it so it doesn't bleed into the forecast plot

                    # --- GENERATE FORECAST ---
                    conditional_forecast_draws = dynamic_conditional_forecast(
                        Y_stand[final_cols], lag_val, B_draws, Sigma_draws, h_steps, 
                        future_exog=future_exog, condition_dict=condition_dict
                    )

                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=h_steps+1, freq='QS-MAR')[1:]

                    # --- DISPLAY FORECAST PLOT ---
                    st.subheader("📈 GDP Forecast")
                    
                    # CAPTURE the two returned dataframes
                    summary_df, full_df = plot_pure_forecast(
                        conditional_forecast_draws, future_dates, standardization_dict, 
                        target_var="HKGDP", train_df=df, n_train_tail=16, 
                        interval_width=interval_width # Pass the slider value here!
                    )
                    
                    st.pyplot(plt.gcf())
                    plt.clf() 

                    # --- DOWNLOAD FUNCTIONALITY ---
                    st.subheader("📥 Download Data")
                    download_col1, download_col2 = st.columns(2)
                    
                    # If the user checked the summary box, show the button
                    with download_col1:
                        if dl_summary:
                            st.download_button(
                                label="Download Summary CSV",
                                data=summary_df.to_csv(),
                                file_name=f"HKGDP_Summary_{interval_width}pct.csv",
                                mime="text/csv"
                            )
                    
                    # If the user checked the full draws box, show the button
                    with download_col2:
                        if dl_full:
                            st.download_button(
                                label="Download Full Draws CSV",
                                data=full_df.to_csv(),
                                file_name="HKGDP_Full_Draws.csv",
                                mime="text/csv"
                            )

                st.success("Forecast Generated Successfully!")

            # ==========================================
            # NEW: HISTORICAL OUT-OF-SAMPLE TEST
            # ==========================================
            st.subheader("🔬 Historical Out-of-Sample Test")
            st.markdown("Test how this specification would have performed historically.")
            
            # Create a row: Date picker on left (wider), Button on right (narrower)
            hist_col1, hist_col2 = st.columns([2, 1])
            
            with hist_col1:
                # Default to a date 4-5 years ago as a sensible default
                default_hist_date = pd.to_datetime('2024-01-01')
                hist_cutoff = st.date_input("Training Cutoff Date:", value=default_hist_date, key="hist_date")
            
            with hist_col2:
                # Align the button to the bottom of the column so it lines up with the date input
                st.write("") # Spacer
                st.write("") # Spacer
                hist_button = st.button("Run Historical Test", use_container_width=True)
                
            if hist_button:
                with st.spinner(f"Estimating BVAR using data up to {hist_cutoff.strftime('%b %Y')}..."):
                    
                    # We need to catch the equation print() output again for this separate run
                    hist_equation_buffer = io.StringIO()
                    
                    # Wrap the call to redirect print statements

                    diff_exog_dict = {
                        "HKGDP": ["FFR", "China_PMI_NEO"], "Imports": ["FFR", "China_PMI_NEO"], 
                        "HSI": ["FFR", "China_PMI_NEO"], "PPI": ["FFR", "China_PMI_NEO"], 
                        "Exports": ["FFR", "China_PMI_NEO"], "RSV": ["FFR", "China_PMI_NEO"], 
                        "PST_Volume": ["FFR", "China_PMI_NEO"], "FFR": ["China_PMI_NEO"], 
                        "China_PMI_NEO": ["FFR"]
                    }
                    exog_dict = create_exog_dict(final_cols, diff_exog_dict)

                    with contextlib.redirect_stdout(hist_equation_buffer):
                        hist_coeff_fig = integrated_forecast_graph(
                            df=df,
                            cutoff_date=pd.to_datetime(hist_cutoff), # Convert date input to pandas datetime
                            col_spec=final_cols,
                            p=lag_val,
                            lambda_val=lambda_val,
                            delta=delta,
                            decay=decay,
                            exog_list=selected_exog, # Uses the same exog selection from the sidebar!
                            exog_dict=exog_dict,
                            n_draws=n_draws,
                            h_steps=h_steps,
                            plot_var_idx=target_var_idx if show_plot else None,
                            print_eq_idx=target_var_idx if show_equation else None,
                            condition_dict=condition_dict,
                            include_training=True,
                            future_exog=None
                        )
                    
                    # --- DISPLAY HISTORICAL EQUATION ---
                    if show_equation:
                        hist_eq_text = hist_equation_buffer.getvalue()
                        if hist_eq_text.strip():
                            st.subheader("📝 Historical Test - Estimating Equation")
                            st.code(hist_eq_text, language=None)

                    # --- DISPLAY HISTORICAL COEFFICIENT PLOT ---
                    if show_plot:
                        st.subheader("📊 Historical Test - Posterior Coefficients")
                        st.pyplot(hist_coeff_fig)

                    # --- DISPLAY HISTORICAL FORECAST GRAPH ---
                    st.subheader("📈 Historical Test - Forecast vs Actuals")
                    st.pyplot(plt.gcf())
                    plt.clf()

                st.success("Historical Test Complete!")

        except Exception as e:
            st.error(f"Failed to process file or run model. Error: {e}")
