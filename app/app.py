"""
app/app.py
==========
Main Entrance for Data Mining 3-Task Production System
Author: Mostafa (Team Leader & Integration Engineer)
"""

import streamlit as st

st.set_page_config(
    page_title="Data Mining 3-Task Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Data Mining 3-Task Production Suite")
st.markdown("#### Integrated Machine Learning & Pattern Discovery System")

st.info("""
Welcome to the official Data Mining project portal. This interactive application connects three complete, production-grade tasks:
1. **Binary Classification:** Customer Churn Prediction
2. **Continuous Regression:** Appliances Energy Consumption Forecasting
3. **Pattern Mining:** Market Basket Product Recommendation System
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 1. Classification")
    st.markdown("""
    **Customer Churn Prediction**
    - **Target:** Predict whether a customer leaves or stays.
    - **Dataset:** IBM Telco Customer Churn.
    - **Algorithms:** Logistic Regression, Random Forest, XGBoost.
    - **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.
    """)
    st.caption("👈 Select 'Classification' from the sidebar to test predictions.")

with col2:
    st.subheader("⚡ 2. Regression")
    st.markdown("""
    **Energy Consumption Forecasting**
    - **Target:** Continuous energy usage in Watt-hours (Wh).
    - **Dataset:** UCI Appliances Energy Prediction.
    - **Algorithms:** Linear Regression, Random Forest, XGBoost.
    - **Metrics:** MSE, RMSE, R² Score, Residual Analysis.
    """)
    st.caption("👈 Select 'Regression' from the sidebar to test predictions.")

with col3:
    st.subheader("🛒 3. Association Rules")
    st.markdown("""
    **Product Recommendation System**
    - **Target:** Item-to-item recommendations from market baskets.
    - **Dataset:** UCI Online Retail Dataset.
    - **Algorithms:** Apriori vs. FP-Growth.
    - **Metrics:** Support, Confidence, Lift.
    """)
    st.caption("👈 Select 'Association' from the sidebar to explore recommendations.")

st.divider()

st.subheader("👥 Project Engineering Team")
team_members = [
    ("Mostafa", "Project Leader & Integration Lead", "Architecture, Git Governance, PR Reviews, Streamlit App, Final Delivery"),
    ("Febronia", "Classification Data Engineer", "Telco Churn EDA, Data Cleaning, Missing Values, Imbalance (SMOTE), Scaling"),
    ("Rovan", "Classification Modeling Engineer", "Logistic Regression, Random Forest, XGBoost, 6 Evaluation Metrics, ROC-AUC"),
    ("Monzer", "Regression Data Engineer", "Appliances Energy EDA, Outliers, Feature Engineering, Chronological Split"),
    ("Ebrahim", "Regression Modeling Engineer", "Linear Regression, Random Forest, XGBoost Regressor, MSE/RMSE, R², Residuals"),
    ("Basmala", "Association Data Engineer", "Online Retail Cleaning, Returns/Cancellations Removal, One-Hot Basket Matrix"),
    ("Marwa", "Association Mining Engineer", "Apriori vs. FP-Growth, Rule Mining (Support, Confidence, Lift), Recommender Engine")
]

cols = st.columns(len(team_members))
for col, (name, role, desc) in zip(cols, team_members):
    with col:
        st.markdown(f"**{name}**")
        st.caption(f"*{role}*")
        st.write(desc)
