"""
src/association/preprocessing.py
================================
Association Rules Preprocessing & Basket Matrix Construction
Owner: Basmala
Contract Validator: src.contracts.validate_basket_matrix
"""

import pandas as pd
import numpy as np

def build_basket_matrix(cleaned_data_path: str, country: str = 'United Kingdom', min_item_frequency: int = 1) -> pd.DataFrame:
    """
    Transforms raw transaction rows into a 1-Hot Binary Matrix (Invoices x Items).
    """
    df = pd.read_csv(cleaned_data_path)
    
    # 1. Cleaning steps
    df['Description'] = df['Description'].astype(str).str.strip()
    df = df.dropna(subset=['InvoiceNo', 'Description'])
    
    # Remove cancellations and invalid quantities/prices
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    df = df[df['Quantity'] > 0]
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'] > 0]
        
    # Optional country filter
    if country and 'Country' in df.columns and country in df['Country'].unique():
        df = df[df['Country'] == country]
        
    # Filter very rare items
    item_counts = df['Description'].value_counts()
    frequent_items = item_counts[item_counts >= min_item_frequency].index
    df = df[df['Description'].isin(frequent_items)]
    
    # 2. Pivot & Encode Binary Matrix
    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
              
    basket_binary = (basket > 0).astype(bool)
    
    # Filter single-item baskets
    basket_binary = basket_binary[basket_binary.sum(axis=1) >= 2]
    return basket_binary
