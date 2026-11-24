"""
scripts/download_datasets.py
============================
Automated utility to fetch the three public datasets:
1. Telco Customer Churn (IBM / Kaggle) -> datasets/classification/raw_telco_churn.csv
2. Appliances Energy Prediction (UCI)  -> datasets/regression/raw_energy_data.csv
3. Online Retail Dataset (UCI)         -> datasets/association/raw_online_retail.csv
"""

import os
import urllib.request

DATASETS = {
    'classification': {
        'url': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
        'destination': 'datasets/classification/raw_telco_churn.csv',
        'name': 'IBM Telco Customer Churn'
    },
    'regression': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv',
        'destination': 'datasets/regression/raw_energy_data.csv',
        'name': 'UCI Appliances Energy Prediction'
    },
    'association': {
        'url': 'https://archive.ics.uci.edu/static/public/352/online+retail.zip',
        'destination': 'datasets/association/online_retail.zip',
        'name': 'UCI Online Retail'
    }
}

def download_dataset(category: str):
    info = DATASETS.get(category)
    if not info:
        print(f"Unknown category: {category}")
        return
        
    os.makedirs(os.path.dirname(info['destination']), exist_ok=True)
    if os.path.exists(info['destination']):
        print(f"[{info['name']}] already exists at {info['destination']}. Skipping.")
        return
        
    print(f"[{info['name']}] Downloading from {info['url']}...")
    try:
        urllib.request.urlretrieve(info['url'], info['destination'])
        print(f"[{info['name']}] Successfully saved to {info['destination']}")
    except Exception as e:
        print(f"[{info['name']}] Download failed: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("Data Mining 3-Task Production Suite - Dataset Downloader")
    print("=" * 60)
    for cat in DATASETS:
        download_dataset(cat)
    print("Done!")
