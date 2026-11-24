"""
tests/test_contracts.py
========================
Automated validation of all integration contracts and mock workflows.
"""

import pytest
import os
try:
    import pandas as pd
except ImportError:
    pd = None
from src.contracts import (
    validate_contracts,
    validate_churn_input,
    validate_churn_output,
    validate_energy_input,
    validate_energy_output,
    validate_recommendation_input,
    validate_recommendation_output,
    validate_basket_matrix
)
from src.classification.model import predict_churn
from src.regression.model import predict_energy
from src.association.recommendations import recommend_products

def test_full_contracts_self_check():
    """Runs the master contract validation routine."""
    assert validate_contracts() is True

def test_classification_mock_inference():
    """Verifies that predict_churn satisfies the classification contract."""
    sample = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 24, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.5, 'TotalCharges': 2292.0
    }
    res = predict_churn(sample)
    assert validate_churn_output(res) is True
    assert res['churn_flag'] in (0, 1)

def test_regression_mock_inference():
    """Verifies that predict_energy satisfies the regression contract."""
    sample = {
        'T1': 21.0, 'RH_1': 42.0, 'T2': 20.0, 'RH_2': 40.0, 'T3': 22.0, 'RH_3': 41.0,
        'T4': 20.0, 'RH_4': 39.0, 'T5': 19.0, 'RH_5': 45.0, 'T6': 8.0, 'RH_6': 80.0,
        'T7': 19.0, 'RH_7': 35.0, 'T8': 21.0, 'RH_8': 42.0, 'T9': 20.0, 'RH_9': 38.0,
        'T_out': 7.5, 'Press_mm_hg': 750.0, 'RH_out': 85.0, 'Windspeed': 4.0,
        'Visibility': 40.0, 'Tdewpoint': 5.0, 'rv1': 25.0, 'rv2': 25.0,
        'hour': 18, 'day_of_week': 2
    }
    res = predict_energy(sample)
    assert validate_energy_output(res) is True
    assert res['predicted_energy_Wh'] >= 0.0

def test_recommendation_mock_inference():
    """Verifies that recommend_products satisfies the association rules contract."""
    basket = ['WHITE HANGING HEART T-LIGHT HOLDER']
    recs = recommend_products(basket, top_n=3)
    assert validate_recommendation_output(recs) is True
    assert len(recs) > 0
    assert recs[0]['recommended_item'] not in basket

def test_mock_basket_matrix_generation():
    """Tests basket matrix creation using mock retail data."""
    if pd is None:
        pytest.skip("pandas not installed in current environment; skipping dataframe test")
    from src.association.preprocessing import build_basket_matrix
    mock_retail_path = os.path.join(os.path.dirname(__file__), 'mocks/mock_retail.csv')
    basket_df = build_basket_matrix(mock_retail_path, country='United Kingdom', min_item_frequency=1)
    assert validate_basket_matrix(basket_df) is True
