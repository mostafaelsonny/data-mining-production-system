"""
app/pages/1_Classification_Churn.py
===================================
Classification Dashboard: Customer Churn Prediction
"""

import streamlit as st
import os
import pandas as pd
from src.classification.model import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🎯", layout="wide")

st.title("🎯 Customer Churn Prediction")
st.markdown("Predict whether a telecom subscriber will leave the service or stay.")

tab1, tab2 = st.tabs(["🔮 Real-Time Inference (New Predictions)", "📊 Evaluation & Visualizations"])

with tab1:
    st.subheader("Enter Customer Profile")
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months with Company)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=65.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly * max(1, tenure)))

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        input_data = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection, 'TechSupport': support,
            'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
        }
        res = predict_churn(input_data)
        
        st.divider()
        if res['churn_flag'] == 1:
            st.error(f"⚠️ **Result:** {res['prediction']} (Risk Probability: {res['churn_probability']:.1%})")
        else:
            st.success(f"✅ **Result:** {res['prediction']} (Retention Probability: {1.0 - res['churn_probability']:.1%})")
        st.caption(f"Inference Engine: {res['model_used']}")

with tab2:
    st.subheader("Model Evaluation Metrics & Artifacts")
    metrics_path = "results/classification/classification_metrics.csv"
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.dataframe(m_df, use_container_width=True)
    else:
        st.info("Metrics table will be rendered here upon model training completion (Day 4).")
        
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix**")
        cm_path = "results/classification/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path)
        else:
            st.caption("Waiting for confusion_matrix.png")
            
    with col_b:
        st.markdown("**ROC-AUC Curve**")
        roc_path = "results/classification/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.caption("Waiting for roc_curve.png")

    comp_csv = "results/classification/model_comparison.csv"
    if os.path.exists(comp_csv):
        st.divider()
        st.markdown("**Model Comparison (5-Fold Stratified CV & Held-Out Test)**")
        st.dataframe(pd.read_csv(comp_csv), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        pva_path = "results/classification/predicted_vs_actual.png"
        if os.path.exists(pva_path):
            st.markdown("**Predicted vs. Actual Results**")
            st.image(pva_path)
    with col_d:
        fi_path = "results/classification/feature_importance.png"
        if os.path.exists(fi_path):
            st.markdown("**Top Churn Drivers (Feature Importance)**")
            st.image(fi_path)
