"""
src/association/preprocessing.py
================================
Association Rules Preprocessing & Basket Matrix Construction
Owner: Basmala
Contract Validator: src.contracts.validate_basket_matrix
"""

import pandas as pd
import numpy as np

NON_PRODUCT_CODES = ['POST', 'DOT', 'M', 'D', 'C2', 'BANK CHARGES',
                     'AMAZONFEE', 'S', 'B', 'CRUK', 'PADS']
NON_PRODUCT_DESC = ['POSTAGE', 'DOTCOM POSTAGE', 'MANUAL', 'BANK CHARGES',
                    'DISCOUNT', 'AMAZON FEE', 'CARRIAGE', 'SAMPLES',
                    'CRUK COMMISSION', 'ADJUST BAD DEBT']


def clean_retail_data(raw_df_or_path) -> pd.DataFrame:
    """
    Cleans UCI Online Retail transactions.
    Accepts a DataFrame (raw or already cleaned) or a path to a raw CSV file.
    """
    if isinstance(raw_df_or_path, pd.DataFrame):
        df = raw_df_or_path.copy()
    else:
        df = pd.read_csv(raw_df_or_path, encoding='ISO-8859-1',
                         dtype={'InvoiceNo': str, 'StockCode': str})

    # 1. Remove rows without a product description
    df['Description'] = df['Description'].str.strip()
    df = df.dropna(subset=['InvoiceNo', 'Description'])

    # 2. Remove cancelled invoices (start with C)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]

    # 3. Remove invalid quantities and prices
    df = df[df['Quantity'] > 0]
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'] > 0]

    # 4. Standardize InvoiceNo format
    df['InvoiceNo'] = df['InvoiceNo'].str.strip()
    df['InvoiceNo'] = df['InvoiceNo'].str.extract(r'([CA]?\d{6})\s*$')[0]
    df = df.dropna(subset=['InvoiceNo'])

    # 5. Remove non-product rows (postage, fees, discounts, ...)
    if 'StockCode' in df.columns:
        df['StockCode'] = df['StockCode'].astype(str).str.strip()
        df = df[~df['StockCode'].str.upper().isin(NON_PRODUCT_CODES)]
    df = df[~df['Description'].str.upper().isin(NON_PRODUCT_DESC)]

    # 6. Remove bad-debt invoices (start with A)
    df = df[~df['InvoiceNo'].str.startswith('A')]
    return df


def build_basket_matrix(cleaned_data_path: str, country: str = 'United Kingdom', min_item_frequency: int = 15) -> pd.DataFrame:
    """
    Transforms raw or cleaned transaction rows into a 1-Hot Binary Matrix (Invoices x Items).
    """
    # 1. Cleaning steps (same logic as notebook 05)
    df = clean_retail_data(cleaned_data_path)

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


def save_basket(basket_df: pd.DataFrame, output_path: str):
    """
    Saves the basket matrix to disk as a pickle file.
    """
    basket_df.to_pickle(output_path)
    print(f"Basket Matrix successfully saved to {output_path}")