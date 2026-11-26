"""
src/classification/evaluation.py
================================
Classification Evaluation Metrics & Visualization Suite (Customer Churn)
Owner: Rovan
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_confusion_matrix


def plot_roc_curves(roc_entries: list, output_path: str):
    """
    Plots ROC-AUC curves for one or more evaluated classification models.
    `roc_entries` is a list of tuples: (model_name, y_true, y_prob).
    """
    plt.figure(figsize=(7, 5))
    colors = ['darkorange', 'teal', 'crimson', 'royalblue']
    for idx, (name, y_true, y_prob) in enumerate(roc_entries):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        lw = 2.5 if idx == 0 else 1.8
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.2, label='Random Baseline (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC-AUC Characteristic Curve (Customer Churn)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.25)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test, y_pred, y_prob, output_path: str, model_name: str = "Champion Model"):
    """
    Visualizes Predicted vs. Actual results after model execution:
    - Left panel: Bar comparison of Actual vs Predicted class counts.
    - Right panel: Predicted Churn Probability distribution split by Actual label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel 1: Class Count Comparison
    actual_counts = pd.Series(y_test).value_counts().reindex([0, 1], fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)
    comp_df = pd.DataFrame({
        'Actual': [actual_counts[0], actual_counts[1]],
        'Predicted': [pred_counts[0], pred_counts[1]]
    }, index=['Stay (0)', 'Churn (1)'])

    comp_df.plot(kind='bar', ax=axes[0], color=['#2b7bba', '#e06666'], edgecolor='black', width=0.65)
    axes[0].set_title(f'Actual vs. Predicted Class Counts\n({model_name})')
    axes[0].set_ylabel('Number of Customers')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', alpha=0.25)
    for container in axes[0].containers:
        axes[0].bar_label(container, padding=2, fontsize=9)

    # Panel 2: Probability Separation by Actual Class
    prob_df = pd.DataFrame({'Actual': np.asarray(y_test), 'Churn_Prob': np.asarray(y_prob)})
    sns.kdeplot(
        data=prob_df[prob_df['Actual'] == 0], x='Churn_Prob',
        fill=True, common_norm=False, alpha=0.45, color='#2b7bba', label='Actual Stay (0)', ax=axes[1]
    )
    sns.kdeplot(
        data=prob_df[prob_df['Actual'] == 1], x='Churn_Prob',
        fill=True, common_norm=False, alpha=0.45, color='#e06666', label='Actual Churn (1)', ax=axes[1]
    )
    axes[1].axvline(0.5, color='black', linestyle='--', lw=1.2, label='Decision Threshold (0.50)')
    axes[1].set_title('Predicted Churn Probability vs. Actual Outcome')
    axes[1].set_xlabel('Predicted Churn Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_feature_importance(pipeline, output_path: str, top_n: int = 15):
    """
    Extracts and plots top feature importances (or absolute coefficients)
    from the fitted classification pipeline.
    """
    preprocessor = pipeline.named_steps.get('preprocessor')
    classifier = pipeline.named_steps.get('classifier')
    if preprocessor is None or classifier is None:
        return None

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return None

    cleaned_names = [
        name.replace('num__', '').replace('cat__', '') for name in feature_names
    ]

    if hasattr(classifier, 'feature_importances_'):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        metric_label = 'Feature Importance Score'
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(np.asarray(classifier.coef_[0], dtype=float))
        metric_label = 'Absolute Coefficient Magnitude'
    else:
        return None

    fi_df = pd.DataFrame({
        'Feature': cleaned_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 5.5))
    sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {len(fi_df)} Features Driving Customer Churn ({type(classifier).__name__})')
    plt.xlabel(metric_label)
    plt.ylabel('Feature')
    plt.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fi_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """
    Plots Test Set metric comparison across trained classification models.
    """
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_cols = [c for c in metric_cols if c in comparison_df.columns]
    if not available_cols:
        return

    melted = comparison_df.melt(
        id_vars=['Model'],
        value_vars=available_cols,
        var_name='Metric',
        value_name='Score'
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='Set2', edgecolor='black')
    plt.title('Classification Model Comparison on Held-Out Test Set')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.25)
    plt.legend(title='Model', loc='lower right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def evaluate_classification(
    pipeline_path: str,
    X_test,
    y_test,
    results_dir: str = 'results/classification',
    fitted_models: dict = None,
    comparison_df: pd.DataFrame = None
) -> dict:
    """
    Computes all 6 mandatory classification evaluation metrics, saves CSV reports,
    and generates Confusion Matrix, ROC-AUC, Predicted vs. Actual, and Feature Importance plots.
    """
    os.makedirs(results_dir, exist_ok=True)
    pipeline = joblib.load(pipeline_path)
    model_name = type(pipeline.named_steps.get('classifier', pipeline)).__name__

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics_record = {
        'Model': model_name,
        **{k: round(float(v), 4) for k, v in metrics.items()},
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }
    metrics_df = pd.DataFrame([metrics_record])
    metrics_df.to_csv(os.path.join(results_dir, 'classification_metrics.csv'), index=False)

    # 1. Confusion Matrix
    plot_confusion_matrix(
        cm,
        class_names=['Stay (0)', 'Churn (1)'],
        output_path=os.path.join(results_dir, 'confusion_matrix.png'),
        title=f'Churn Confusion Matrix ({model_name})'
    )

    # 2. ROC-AUC Curve (Champion + peers if provided)
    roc_entries = []
    if fitted_models:
        for name, mdl in fitted_models.items():
            roc_entries.append((name, y_test, mdl.predict_proba(X_test)[:, 1]))
    else:
        roc_entries.append((model_name, y_test, y_prob))
    plot_roc_curves(roc_entries, os.path.join(results_dir, 'roc_curve.png'))

    # 3. Predicted vs Actual Results Visualization (Rubric Requirement E)
    plot_predicted_vs_actual(
        y_test, y_pred, y_prob,
        output_path=os.path.join(results_dir, 'predicted_vs_actual.png'),
        model_name=model_name
    )

    # 4. Feature Importance Visualization
    plot_feature_importance(
        pipeline,
        output_path=os.path.join(results_dir, 'feature_importance.png')
    )

    # 5. Model Comparison Chart
    if comparison_df is not None and not comparison_df.empty:
        plot_model_comparison(
            comparison_df,
            output_path=os.path.join(results_dir, 'model_comparison.png')
        )

    return {
        **metrics,
        'Confusion Matrix': cm.tolist(),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }
