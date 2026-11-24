"""
src/classification/model.py
===========================
Classification Modeling, Evaluation & Inference (Customer Churn)
Owner: Rovan
Contract Validator: src.contracts.validate_churn_output
"""

import os
try:
    import joblib
except ImportError:
    joblib = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
except ImportError:
    np = None
from src.contracts import validate_churn_input, validate_churn_output

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/classification/churn_pipeline.pkl')

def predict_churn(customer_data: dict) -> dict:
    """
    Public inference function satisfying Contract #1.
    Receives raw customer dictionary, runs validation, executes pipeline prediction,
    and returns standardized result.
    """
    # 1. Enforce Input Contract
    validated_data = validate_churn_input(customer_data)
    
    # 2. Check if model artifact exists, else run fallback mock
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([validated_data])
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]
        model_name = type(pipeline.named_steps.get('classifier', pipeline)).__name__
    else:
        # Graceful development mock for Day 1
        tenure = float(validated_data.get('tenure', 12))
        monthly = float(validated_data.get('MonthlyCharges', 70))
        # Simple heuristic probability for mock testing
        prob = min(max(0.05, (monthly / 120.0) * (1.0 / (1.0 + tenure / 10.0))), 0.95)
        pred = 1 if prob >= 0.5 else 0
        model_name = "Baseline Heuristic (Mock)"
        
    result = {
        'prediction': 'Customer Likely to Churn' if pred == 1 else 'Customer Likely to Stay',
        'churn_probability': float(prob),
        'churn_flag': int(pred),
        'model_used': model_name
    }
    
    # 3. Enforce Output Contract
    validate_churn_output(result)
    return result
