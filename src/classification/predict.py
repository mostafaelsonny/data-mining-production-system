"""
src/classification/predict.py
=============================
New Customer Churn Prediction Engine (Contract-Validated)
Owner: Rovan
"""

from src.classification.model import predict_churn


def predict_new_customer(customer_dict: dict) -> dict:
    """
    Predicts churn risk and probability for a single new customer profile.
    Delegates to `predict_churn` so input/output integration contracts
    in `src/contracts.py` are strictly enforced.
    """
    return predict_churn(customer_dict)
