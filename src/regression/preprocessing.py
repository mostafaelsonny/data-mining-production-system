"""
src/regression/preprocessing.py
===============================
Regression Preprocessing Pipeline (Energy Consumption)
Owner: Monzer
Contract Validator: src.contracts.validate_energy_input
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclic/temporal features from date column: hour and day of week.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df = df.drop(columns=['date'])
    return df

def prepare_energy_data(filepath: str, test_ratio: float = 0.2):
    """
    Preprocesses Appliances Energy dataset with strict Chronological Train/Test split.
    """
    df = pd.read_csv(filepath)
    df = extract_temporal_features(df)
    
    target_col = 'Appliances'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {filepath}")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    
    # Strictly Chronological Split (No Random Shuffling)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names
