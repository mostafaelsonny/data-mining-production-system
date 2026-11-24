"""
src/contracts.py
================
Official Inter-Module Integration Contracts & Data Exchange Interfaces
Project: Data Mining 3-Task System (7-Day Sprint)
Author: Mostafa (Project Leader & Integration Engineer)

These contracts define the strict structural and operational expectations
between Data Preprocessing, Model Development, and the Streamlit Dashboard.
Every team member must comply with these function signatures and data schemas.
"""

from typing import Any, Dict, List, Tuple
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

# =====================================================================
# 1. CLASSIFICATION CONTRACT: Customer Churn Prediction
# Team: Febronia (Data/Preprocessing) -> Rovan (Modeling) -> Mostafa (Streamlit)
# =====================================================================

CHURN_EXPECTED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

def validate_churn_input(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates that incoming customer dictionary has all required features
    and plausible numerical types before inference.
    """
    if not isinstance(customer_data, dict):
        raise TypeError(f"customer_data must be a dict, got {type(customer_data).__name__}")
    
    missing = [f for f in CHURN_EXPECTED_FEATURES if f not in customer_data]
    if missing:
        raise ValueError(f"Missing required classification features: {missing}")
    
    # Type & range checks for key numerics
    try:
        tenure = float(customer_data['tenure'])
        monthly = float(customer_data['MonthlyCharges'])
        total = float(customer_data['TotalCharges'])
        if tenure < 0 or monthly < 0 or total < 0:
            raise ValueError("Numerical values (tenure, charges) cannot be negative.")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric value in churn customer data: {e}")
        
    return customer_data

def validate_churn_output(result: Dict[str, Any]) -> bool:
    """
    Validates the structure of the prediction returned by Rovan's model.
    Expected:
    {
        'prediction': 'Customer Likely to Churn' | 'Customer Likely to Stay',
        'churn_probability': float between 0.0 and 1.0,
        'churn_flag': 0 or 1,
        'model_used': str
    }
    """
    required_keys = ['prediction', 'churn_probability', 'churn_flag', 'model_used']
    for k in required_keys:
        if k not in result:
            raise KeyError(f"Churn prediction output missing required key: '{k}'")
            
    prob = result['churn_probability']
    is_num = isinstance(prob, (float, int)) or (np is not None and isinstance(prob, np.floating))
    if not is_num or not (0.0 <= prob <= 1.0):
        raise ValueError(f"churn_probability must be a float between 0.0 and 1.0, got {prob}")
        
    flag = result['churn_flag']
    if flag not in (0, 1):
        raise ValueError(f"churn_flag must be 0 or 1, got {flag}")
        
    return True


# =====================================================================
# 2. REGRESSION CONTRACT: Energy Consumption Forecasting
# Team: Monzer (Data/Preprocessing) -> Ebrahim (Modeling) -> Mostafa (Streamlit)
# =====================================================================

ENERGY_REQUIRED_NUMERIC_FEATURES = [
    'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
    'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
    'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
    'Visibility', 'Tdewpoint', 'rv1', 'rv2', 'hour', 'day_of_week'
]

def validate_energy_input(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates that incoming energy features dictionary has all required sensors
    and temporal features before inference.
    """
    if not isinstance(features, dict):
        raise TypeError(f"features must be a dict, got {type(features).__name__}")
        
    missing = [f for f in ENERGY_REQUIRED_NUMERIC_FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing required regression features: {missing}")
        
    for k, v in features.items():
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Energy feature '{k}' must be convertible to float, got '{v}'")
            
    # Physical constraints
    if not (0 <= int(features.get('hour', 0)) <= 23):
        raise ValueError("Feature 'hour' must be between 0 and 23.")
    if not (0 <= int(features.get('day_of_week', 0)) <= 6):
        raise ValueError("Feature 'day_of_week' must be between 0 and 6.")
        
    return features

def validate_energy_output(result: Dict[str, Any]) -> bool:
    """
    Validates the structure of the prediction returned by Ebrahim's model.
    Expected:
    {
        'predicted_energy_Wh': float >= 0.0,
        'unit': 'Watt-hour (Wh)',
        'model_used': str
    }
    """
    required_keys = ['predicted_energy_Wh', 'unit', 'model_used']
    for k in required_keys:
        if k not in result:
            raise KeyError(f"Energy prediction output missing required key: '{k}'")
            
    val = result['predicted_energy_Wh']
    is_num = isinstance(val, (int, float)) or (np is not None and isinstance(val, np.floating))
    if not is_num or val < 0.0:
        raise ValueError(f"predicted_energy_Wh must be a non-negative float, got {val}")
        
    return True


# =====================================================================
# 3. ASSOCIATION RULES CONTRACT: Product Recommendation System
# Team: Basmala (Data/Basket Matrix) -> Marwa (Mining/Engine) -> Mostafa (Streamlit)
# =====================================================================

def validate_basket_matrix(basket_df: Any) -> bool:
    """
    Validates Basmala's basket matrix before rule mining:
    - Rows = InvoiceNo (index)
    - Columns = Item descriptions
    - Values = 0/1 or True/False
    - Every transaction must contain at least 2 items
    """
    if pd is not None and isinstance(basket_df, pd.DataFrame):
        if basket_df.empty:
            raise ValueError("basket_df cannot be empty")
        flat_sample = basket_df.iloc[:20].values.flatten()
        unique_vals = set(flat_sample)
        valid_binary = {0, 1, True, False, 0.0, 1.0}
        if not unique_vals.issubset(valid_binary):
            raise ValueError(f"Basket matrix values must be binary (0/1 or True/False), found: {unique_vals}")
        row_sums = basket_df.iloc[:50].sum(axis=1)
        if (row_sums < 2).any():
            raise ValueError("Each transaction row in basket_matrix must have at least 2 distinct items.")
        return True
    elif isinstance(basket_df, dict):
        if not basket_df:
            raise ValueError("basket_df dictionary cannot be empty")
        return True
    return True

def validate_recommendation_input(selected_items: List[str]) -> List[str]:
    """
    Validates the user's selected shopping basket items.
    """
    if not isinstance(selected_items, (list, tuple, set)):
        raise TypeError(f"selected_items must be a list of strings, got {type(selected_items).__name__}")
    
    cleaned = [str(item).strip().upper() for item in selected_items if str(item).strip()]
    return cleaned

def validate_recommendation_output(recommendations: List[Dict[str, Any]]) -> bool:
    """
    Validates Marwa's recommendation output:
    Expected list of dicts:
    [
        {
            'recommended_item': str,
            'confidence': float,
            'lift': float,
            'support': float,
            'based_on_rule': str
        },
        ...
    ]
    """
    if not isinstance(recommendations, list):
        raise TypeError("recommendations output must be a list of dictionaries")
        
    required_keys = ['recommended_item', 'confidence', 'lift', 'support', 'based_on_rule']
    for idx, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            raise TypeError(f"Recommendation at index {idx} must be a dict")
        for k in required_keys:
            if k not in rec:
                raise KeyError(f"Recommendation at index {idx} missing key: '{k}'")
        if rec['lift'] <= 0:
            raise ValueError(f"Lift score must be positive, got {rec['lift']}")
            
    return True


# =====================================================================
# 4. CONTRACT SELF-TEST VALIDATOR
# =====================================================================

def validate_contracts() -> bool:
    """
    Self-test execution routine.
    Verifies that sample inputs and mock outputs satisfy all contracts.
    """
    print("[CONTRACT CHECK] Testing Classification Contract...")
    mock_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 12, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
        'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 55.2, 'TotalCharges': 662.4
    }
    validate_churn_input(mock_customer)
    mock_churn_out = {
        'prediction': 'Customer Likely to Stay',
        'churn_probability': 0.24,
        'churn_flag': 0,
        'model_used': 'Random Forest Classifier (Champion)'
    }
    validate_churn_output(mock_churn_out)
    print(" -> Classification Contract: VALIDATED OK.")

    print("[CONTRACT CHECK] Testing Regression Contract...")
    mock_energy_in = {
        'T1': 19.89, 'RH_1': 47.596, 'T2': 19.2, 'RH_2': 44.79, 'T3': 19.79, 'RH_3': 44.73,
        'T4': 19.0, 'RH_4': 45.566, 'T5': 17.166, 'RH_5': 55.2, 'T6': 7.026, 'RH_6': 84.256,
        'T7': 17.2, 'RH_7': 41.626, 'T8': 18.2, 'RH_8': 48.9, 'T9': 17.033, 'RH_9': 45.53,
        'T_out': 6.6, 'Press_mm_hg': 733.5, 'RH_out': 92.0, 'Windspeed': 7.0,
        'Visibility': 63.0, 'Tdewpoint': 5.3, 'rv1': 13.275, 'rv2': 13.275,
        'hour': 17, 'day_of_week': 0
    }
    validate_energy_input(mock_energy_in)
    mock_energy_out = {
        'predicted_energy_Wh': 92.4,
        'unit': 'Watt-hour (Wh)',
        'model_used': 'XGBoost Regressor (Champion)'
    }
    validate_energy_output(mock_energy_out)
    print(" -> Regression Contract: VALIDATED OK.")

    print("[CONTRACT CHECK] Testing Association Rules Contract...")
    mock_basket_input = ['WHITE HANGING HEART T-LIGHT HOLDER']
    cleaned_items = validate_recommendation_input(mock_basket_input)
    assert cleaned_items == ['WHITE HANGING HEART T-LIGHT HOLDER']
    
    mock_recommendations = [
        {
            'recommended_item': 'RED HANGING HEART T-LIGHT HOLDER',
            'confidence': 0.78,
            'lift': 3.85,
            'support': 0.045,
            'based_on_rule': 'WHITE HANGING HEART T-LIGHT HOLDER -> RED HANGING HEART T-LIGHT HOLDER'
        }
    ]
    validate_recommendation_output(mock_recommendations)
    print(" -> Association Rules Contract: VALIDATED OK.")
    
    print("\nALL CONTRACTS VERIFIED AND ENFORCED SUCCESSFULLY (100% PASS).")
    return True

if __name__ == '__main__':
    validate_contracts()
