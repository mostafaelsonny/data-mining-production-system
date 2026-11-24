"""
src/utils/metrics.py
====================
Shared evaluation metric calculators and report formatters.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error, r2_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1-Score': float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        try:
            metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['ROC-AUC'] = np.nan
    return metrics

def compute_regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
