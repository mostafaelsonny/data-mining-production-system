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


def build_candidate_models(random_state: int = 42) -> dict:
    """
    Returns the three candidate classifiers for Churn Prediction:
    1. Logistic Regression (interpretable linear baseline)
    2. Random Forest Classifier (bagging ensemble)
    3. XGBoost Classifier (gradient boosting)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, eval_metric='logloss', n_jobs=-1
        )
    }


def train_and_compare_models(
    data_path: str = 'datasets/classification/raw_telco_churn.csv',
    model_output_path: str = 'models/classification/churn_pipeline.pkl',
    preprocessor_output_path: str = 'models/classification/preprocessor.pkl',
    results_dir: str = 'results/classification'
):
    """
    Trains and compares Logistic Regression, Random Forest, and XGBoost using
    5-Fold Stratified Cross-Validation inside an ImbPipeline (Zero Data Leakage).
    Selects the champion model by 5-Fold Mean F1-Score, fits on full training set,
    evaluates on held-out test set, and persists artifacts.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from src.classification.preprocessing import load_and_clean_raw, get_preprocessor, NUMERIC_FEATURES
    from src.utils.metrics import compute_classification_metrics

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = load_and_clean_raw(data_path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Stratified 80/20 Train/Test Split before any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_candidate_models(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    comparison_rows = []
    fitted_models = {}
    best_cv_f1 = -1.0
    best_model_name = None
    best_pipeline = None

    for name, clf in models.items():
        # ImbPipeline ensures preprocessor & SMOTE fit strictly on each CV training fold
        pipeline = ImbPipeline(steps=[
            ('preprocessor', get_preprocessor(numeric_cols, categorical_cols)),
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_f1 = float(np.mean(cv_results['test_f1']))
        cv_auc = float(np.mean(cv_results['test_roc_auc']))

        # Fit on full training split to record held-out test metrics as well
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        y_pred_test = pipeline.predict(X_test)
        y_prob_test = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(y_test, y_pred_test, y_prob_test)

        row = {
            'Model': name,
            'CV_5Fold_Accuracy': round(float(np.mean(cv_results['test_accuracy'])), 4),
            'CV_5Fold_Precision': round(float(np.mean(cv_results['test_precision'])), 4),
            'CV_5Fold_Recall': round(float(np.mean(cv_results['test_recall'])), 4),
            'CV_5Fold_F1': round(cv_f1, 4),
            'CV_5Fold_ROC_AUC': round(cv_auc, 4),
            'Accuracy': round(test_metrics['Accuracy'], 4),
            'Precision': round(test_metrics['Precision'], 4),
            'Recall': round(test_metrics['Recall'], 4),
            'F1-Score': round(test_metrics['F1-Score'], 4),
            'ROC-AUC': round(test_metrics['ROC-AUC'], 4),
        }
        comparison_rows.append(row)
        print(
            f"[{name}] 5-Fold CV F1: {cv_f1:.4f} | CV ROC-AUC: {cv_auc:.4f} | "
            f"Test F1: {test_metrics['F1-Score']:.4f} | Test Recall: {test_metrics['Recall']:.4f} | "
            f"Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}"
        )

        if cv_f1 > best_cv_f1:
            best_cv_f1 = cv_f1
            best_model_name = name
            best_pipeline = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values('CV_5Fold_F1', ascending=False).reset_index(drop=True)
    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    joblib.dump(best_pipeline, model_output_path)
    if preprocessor_output_path:
        os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
        joblib.dump(best_pipeline.named_steps['preprocessor'], preprocessor_output_path)

    print(f"\nChampion Model Selected: {best_model_name} (5-Fold CV F1 = {best_cv_f1:.4f})")
    print(f"Saved pipeline artifact -> {model_output_path}")

    return best_pipeline, best_model_name, X_train, X_test, y_train, y_test, comparison_df, fitted_models


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
