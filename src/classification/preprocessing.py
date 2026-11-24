"""
src/classification/preprocessing.py
===================================
Classification Data Preprocessing Pipeline (Customer Churn)
Owner: Febronia
Contract Validator: src.contracts.validate_churn_input
"""

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

def load_and_clean_raw(filepath: str) -> pd.DataFrame:
    """
    Loads raw Telco Churn CSV, drops non-predictive customerID,
    fixes whitespace issue in TotalCharges, and maps Churn to 0/1.
    """
    df = pd.read_csv(filepath)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

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

def prepare_churn_data(filepath: str, apply_smote: bool = True):
    """
    Complete preprocessing pipeline following the Zero-Leakage principle.
    """
    df = load_and_clean_raw(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    # Stratified split to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # SMOTE only on training set
    if apply_smote and SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
