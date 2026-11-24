"""
src/regression/model.py
=======================
Regression Modeling, Evaluation & Inference (Energy Consumption)
Owner: Ebrahim
Contract Validator: src.contracts.validate_energy_output
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
from src.contracts import validate_energy_input, validate_energy_output

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/regression/energy_pipeline.pkl')

def predict_energy(features: dict) -> dict:
    """
    Public inference function satisfying Contract #2.
    Receives raw sensor dictionary, validates inputs, executes pipeline,
    and returns non-negative predicted energy consumption.
    """
    # 1. Enforce Input Contract
    validated_features = validate_energy_input(features)
    
    # 2. Check if model artifact exists, else run fallback mock
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([validated_features])
        pred = pipeline.predict(input_df)[0]
        model_name = type(pipeline.named_steps.get('regressor', pipeline)).__name__
    else:
        # Graceful development mock for Day 1
        t1 = float(validated_features.get('T1', 20.0))
        hour = float(validated_features.get('hour', 12))
        # Simple baseline energy estimate (Wh)
        pred = max(20.0, 50.0 + (t1 * 1.5) + (25.0 if 17 <= hour <= 21 else 5.0))
        model_name = "Baseline Empirical (Mock)"
        
    result = {
        'predicted_energy_Wh': round(max(0.0, float(pred)), 2),
        'unit': 'Watt-hour (Wh)',
        'model_used': model_name
    }
    
    # 3. Enforce Output Contract
    validate_energy_output(result)
    return result
