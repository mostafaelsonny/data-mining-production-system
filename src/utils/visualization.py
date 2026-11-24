"""
src/utils/visualization.py
==========================
Publication-quality plotting functions for evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, output_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_actual_vs_predicted(y_actual, y_pred, output_path: str, r2_score_val: float):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_actual, y_pred, alpha=0.3, color='royalblue', edgecolors='k', s=20)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Energy Consumption (Wh)')
    plt.ylabel('Predicted Energy Consumption (Wh)')
    plt.title(f'Actual vs. Predicted Energy (R² = {r2_score_val:.3f})')
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
