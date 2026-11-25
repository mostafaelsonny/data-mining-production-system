"""
src/classification/preprocessing.py
===================================
Classification Data Preprocessing Pipeline (Customer Churn)
Owner: Febronia
Contract Validator: src.contracts.validate_churn_input
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

def load_and_clean_raw(filepath: str) -> pd.DataFrame:
    """
    Loads raw Telco Churn CSV, removes exact duplicate records, drops
    non-predictive customerID, fixes whitespace issue in TotalCharges,
    and maps Churn to binary 0/1.
    """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates().reset_index(drop=True)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'].astype(str).str.strip().replace('', np.nan),
            errors='coerce'
        )
    if 'Churn' in df.columns and not pd.api.types.is_numeric_dtype(df['Churn']):
        df['Churn'] = df['Churn'].astype(str).str.strip().map({'Yes': 1, 'No': 0}).astype(int)
    return df

def detect_outliers_iqr(df: pd.DataFrame, numeric_features: list = None) -> pd.DataFrame:
    """
    Detects numerical outliers using the 1.5 * IQR rule and returns a summary table.
    """
    cols = numeric_features or [c for c in NUMERIC_FEATURES if c in df.columns]
    records = []
    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        records.append({
            'Feature': col,
            'Q1': round(float(q1), 2),
            'Q3': round(float(q3), 2),
            'IQR': round(float(iqr), 2),
            'Lower_Bound': round(float(lower), 2),
            'Upper_Bound': round(float(upper), 2),
            'Outlier_Count': outlier_count,
            'Outlier_Pct': round(100.0 * outlier_count / max(len(series), 1), 2)
        })
    return pd.DataFrame(records)

def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds the ColumnTransformer for numeric and categorical pipelines.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def prepare_churn_data(
    filepath: str,
    apply_smote: bool = True,
    save_preprocessor_path: str = None
):
    """
    Complete preprocessing pipeline following the Zero-Leakage principle:
    1. Clean raw data & remove duplicates
    2. Stratified 80/20 Train/Test split before any fitting
    3. Fit ColumnTransformer (imputation, scaling, one-hot encoding) on X_train only
    4. Transform X_test using the fitted preprocessor
    5. Apply SMOTE oversampling strictly on X_train_proc, y_train
    """
    df = load_and_clean_raw(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Stratified split to preserve class ratio (~73.5% Stay vs ~26.5% Churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if save_preprocessor_path:
        os.makedirs(os.path.dirname(save_preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, save_preprocessor_path)

    # SMOTE only on training set (Zero Leakage)
    if apply_smote and SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
