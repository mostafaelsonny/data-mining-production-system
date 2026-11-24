"""
app/pages/2_Regression_Energy.py
================================
Regression Dashboard: Energy Consumption Forecasting
"""

import streamlit as st
import os
import pandas as pd
from src.regression.model import predict_energy

st.set_page_config(page_title="Energy Consumption Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ Appliances Energy Consumption Forecasting")
st.markdown("Forecast household appliance energy consumption (Wh) from environmental & temporal sensors.")

tab1, tab2 = st.tabs(["🔮 Real-Time Forecasting (New Predictions)", "📊 Evaluation & Residuals"])

with tab1:
    st.subheader("Sensor & Climate Measurements")
    with st.form("energy_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Indoor Temperatures (°C)**")
            t1 = st.slider("T1 (Kitchen)", 15.0, 30.0, 20.0, 0.1)
            t2 = st.slider("T2 (Living Room)", 15.0, 30.0, 19.5, 0.1)
            t3 = st.slider("T3 (Laundry)", 15.0, 30.0, 20.5, 0.1)
            t4 = st.slider("T4 (Office)", 15.0, 30.0, 19.0, 0.1)
            t5 = st.slider("T5 (Bathroom)", 15.0, 30.0, 18.0, 0.1)
            t6 = st.slider("T6 (Outside North)", -5.0, 30.0, 7.5, 0.1)
            t7 = st.slider("T7 (Ironing Room)", 15.0, 30.0, 18.5, 0.1)
            t8 = st.slider("T8 (Teenager Room)", 15.0, 30.0, 21.0, 0.1)
            t9 = st.slider("T9 (Parents Room)", 15.0, 30.0, 19.0, 0.1)

        with col2:
            st.markdown("**Indoor Humidity (%)**")
            rh1 = st.slider("RH_1 (Kitchen)", 20.0, 90.0, 45.0, 0.5)
            rh2 = st.slider("RH_2 (Living Room)", 20.0, 90.0, 42.0, 0.5)
            rh3 = st.slider("RH_3 (Laundry)", 20.0, 90.0, 44.0, 0.5)
            rh4 = st.slider("RH_4 (Office)", 20.0, 90.0, 43.0, 0.5)
            rh5 = st.slider("RH_5 (Bathroom)", 20.0, 90.0, 52.0, 0.5)
            rh6 = st.slider("RH_6 (Outside North)", 20.0, 100.0, 80.0, 0.5)
            rh7 = st.slider("RH_7 (Ironing Room)", 20.0, 90.0, 38.0, 0.5)
            rh8 = st.slider("RH_8 (Teenager Room)", 20.0, 90.0, 46.0, 0.5)
            rh9 = st.slider("RH_9 (Parents Room)", 20.0, 90.0, 42.0, 0.5)

        with col3:
            st.markdown("**Weather & Temporal Parameters**")
            t_out = st.slider("T_out (Chievres Station °C)", -10.0, 35.0, 7.0, 0.1)
            rh_out = st.slider("RH_out (%)", 20.0, 100.0, 85.0, 0.5)
            press = st.slider("Pressure (mm Hg)", 720.0, 780.0, 750.0, 0.5)
            wind = st.slider("Windspeed (m/s)", 0.0, 20.0, 4.0, 0.1)
            vis = st.slider("Visibility (km)", 1.0, 70.0, 40.0, 1.0)
            tdew = st.slider("Tdewpoint (°C)", -5.0, 20.0, 5.0, 0.1)
            hour = st.slider("Hour of Day (0-23)", 0, 23, 18)
            day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

        submitted = st.form_submit_button("Forecast Energy Consumption", use_container_width=True)

    if submitted:
        input_data = {
            'T1': t1, 'RH_1': rh1, 'T2': t2, 'RH_2': rh2, 'T3': t3, 'RH_3': rh3,
            'T4': t4, 'RH_4': rh4, 'T5': t5, 'RH_5': rh5, 'T6': t6, 'RH_6': rh6,
            'T7': t7, 'RH_7': rh7, 'T8': t8, 'RH_8': rh8, 'T9': t9, 'RH_9': rh9,
            'T_out': t_out, 'Press_mm_hg': press, 'RH_out': rh_out, 'Windspeed': wind,
            'Visibility': vis, 'Tdewpoint': tdew, 'rv1': 20.0, 'rv2': 20.0,
            'hour': hour, 'day_of_week': day
        }
        res = predict_energy(input_data)
        st.divider()
        st.metric(label="Predicted Appliance Consumption", value=f"{res['predicted_energy_Wh']} Wh")
        st.caption(f"Inference Engine: {res['model_used']}")

with tab2:
    st.subheader("Model Evaluation Metrics & Artifacts")
    metrics_path = "results/regression/regression_metrics.csv"
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.dataframe(m_df, use_container_width=True)
    else:
        st.info("Metrics table will be rendered here upon model training completion (Day 4).")
        
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Actual vs. Predicted Consumption Plot**")
        avp_path = "results/regression/actual_vs_predicted.png"
        if os.path.exists(avp_path):
            st.image(avp_path)
        else:
            st.caption("Waiting for actual_vs_predicted.png")
            
    with col_b:
        st.markdown("**Residual Analysis Plot**")
        res_path = "results/regression/residual_plot.png"
        if os.path.exists(res_path):
            st.image(res_path)
        else:
            st.caption("Waiting for residual_plot.png")
