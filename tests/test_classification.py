"""
tests/test_classification.py
============================
Unit and integration tests for Classification (Telco Customer Churn):
- Data cleaning (TotalCharges whitespace -> NaN -> float64, customerID removal)
- Zero-leakage preprocessing (Stratified split, train-only fit & SMOTE)
- Model artifact inference & predict_new_customer contract compliance
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.classification.preprocessing import (
    load_and_clean_raw,
    detect_outliers_iqr,
    prepare_churn_data,
)
from src.classification.model import predict_churn
from src.classification.predict import predict_new_customer
from src.contracts import validate_churn_input, validate_churn_output

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../datasets/classification/raw_telco_churn.csv"
)
MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "mocks/mock_churn.csv")


def test_load_and_clean_raw_fixes_whitespace():
    """Verifies TotalCharges whitespace becomes NaN and Churn maps to 0/1."""
    path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else MOCK_DATA_PATH
    df = load_and_clean_raw(path)
    assert "customerID" not in df.columns
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert set(df["Churn"].unique()).issubset({0, 1})
    if os.path.exists(RAW_DATA_PATH):
        assert df.shape == (7043, 20)
        assert int(df["TotalCharges"].isna().sum()) == 11


def test_detect_outliers_iqr_structure():
    """Verifies IQR outlier summary report."""
    path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else MOCK_DATA_PATH
    df = load_and_clean_raw(path)
    report = detect_outliers_iqr(df)
    assert list(report["Feature"]) == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "Outlier_Count" in report.columns


def test_prepare_churn_data_zero_leakage():
    """Verifies stratified split, zero NaNs, and train-only SMOTE."""
    if not os.path.exists(RAW_DATA_PATH):
        pytest.skip("Raw telco churn dataset not found")

    X_tr, X_te, y_tr, y_te, prep = prepare_churn_data(RAW_DATA_PATH, apply_smote=True)
    assert X_tr.shape[0] == len(y_tr)
    assert X_te.shape[0] == len(y_te)
    assert X_tr.shape[1] == X_te.shape[1]
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()
    # SMOTE balances training set to 50/50 while test set keeps ~26.5% churn ratio
    counts_tr = np.bincount(y_tr)
    assert counts_tr[0] == counts_tr[1]
    test_churn_ratio = float(np.mean(y_te))
    assert 0.25 <= test_churn_ratio <= 0.28


def test_predict_new_customer_contract():
    """Verifies predict_new_customer output matches integration contracts."""
    sample_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.7, "TotalCharges": 151.65,
    }
    validate_churn_input(sample_customer)
    res = predict_new_customer(sample_customer)
    assert validate_churn_output(res) is True
    assert res["prediction"] in ("Customer Likely to Churn", "Customer Likely to Stay")
    assert res["churn_flag"] in (0, 1)
    assert 0.0 <= res["churn_probability"] <= 1.0
