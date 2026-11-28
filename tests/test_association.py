"""
tests/test_association.py
Unit tests for src/association/preprocessing.py
"""
import inspect

import pandas as pd

from src.association.preprocessing import (
    clean_retail_data,
    build_basket_matrix,
    save_basket,
)

COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'Country']


def _sample_raw_df():
    rows = [
        ('536365', '85123A', 'WHITE HANGING HEART', 6, 2.55, 'United Kingdom'),
        ('536365', '71053', 'WHITE METAL LANTERN', 6, 3.39, 'United Kingdom'),
        ('536366', '22633', 'HAND WARMER UNION JACK', 6, 1.85, 'United Kingdom'),  # single-item basket
        ('C536379', '22633', 'HAND WARMER UNION JACK', -1, 1.85, 'United Kingdom'),  # cancelled
        ('536367', '71053', 'WHITE METAL LANTERN', 0, 3.39, 'United Kingdom'),  # quantity 0
        ('536368', '71053', 'WHITE METAL LANTERN', 2, 0.0, 'United Kingdom'),  # price 0
        ('536369', 'POST', 'POSTAGE', 1, 18.0, 'United Kingdom'),  # non-product
        ('536370', '21730', None, 1, 1.0, 'United Kingdom'),  # missing description
        ('A563185', 'B', 'Adjust bad debt', 1, 11062.06, 'United Kingdom'),  # bad debt
        ('536371', '71053', 'WHITE METAL LANTERN', 3, 3.39, 'France'),
        ('536371', '85123A', 'WHITE HANGING HEART', 3, 2.55, 'France'),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_clean_retail_data_removes_invalid_rows():
    cleaned = clean_retail_data(_sample_raw_df())

    assert set(cleaned['InvoiceNo']) == {'536365', '536366', '536371'}
    assert len(cleaned) == 5
    assert (cleaned['Quantity'] > 0).all()
    assert (cleaned['UnitPrice'] > 0).all()
    assert cleaned['Description'].notna().all()
    assert not cleaned['InvoiceNo'].str.startswith(('C', 'A')).any()


def test_build_basket_matrix_is_bool_with_min_two_items(tmp_path):
    csv_path = tmp_path / 'raw.csv'
    _sample_raw_df().to_csv(csv_path, index=False)

    basket = build_basket_matrix(str(csv_path), country='United Kingdom', min_item_frequency=1)

    assert list(basket.index) == ['536365']
    assert (basket.dtypes == bool).all()
    assert (basket.sum(axis=1) >= 2).all()


def test_build_basket_matrix_defaults():
    params = inspect.signature(build_basket_matrix).parameters
    assert params['country'].default == 'United Kingdom'
    assert params['min_item_frequency'].default == 15


def test_save_basket_roundtrip(tmp_path):
    basket = pd.DataFrame({'A': [True, False], 'B': [True, True]}, index=['1', '2'])
    out_path = tmp_path / 'basket.pkl'

    save_basket(basket, str(out_path))

    pd.testing.assert_frame_equal(pd.read_pickle(out_path), basket)