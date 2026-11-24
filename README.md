# Data Mining 3-Task Production System

## 1. Project Overview
This repository contains an end-to-end Data Mining project delivering three distinct machine learning and pattern discovery tasks integrated into a unified interactive Streamlit application:
1. **Classification (Task 4):** Customer Churn Prediction (Binary Classification)
2. **Regression (Task 4):** Energy Consumption Forecasting (Continuous Target)
3. **Association Rules (Task 2):** Product Recommendation System (Market Basket Pattern Mining)

The project adheres to strict Software Engineering standards: modular code structure (`src/`), clean data pipelines with zero data leakage, comprehensive model evaluation, and an interactive presentation interface.

---

## 2. Team Members & Roles
| Member | Assigned Role | Primary Focus |
| :--- | :--- | :--- |
| **Mostafa** | Project Leader & Integration Lead | Architecture, Git Governance, PR Reviews, Streamlit App, Final Delivery |
| **Febronia** | Classification Data Engineer | Telco Churn EDA, Data Cleaning, Missing Values, Imbalance Handling (SMOTE), Scaling |
| **Rovan** | Classification Modeling Engineer | Logistic Regression, Random Forest, XGBoost, 6 Evaluation Metrics, ROC-AUC, Serialization |
| **Monzer** | Regression Data Engineer | Appliances Energy EDA, Outlier Analysis, Feature Engineering, Chronological Split |
| **Ebrahim** | Regression Modeling Engineer | Linear Regression, Random Forest, XGBoost Regressor, MSE/RMSE, R², Residual Analysis |
| **Basmala** | Association Data Engineer | Online Retail Cleaning, Returns/Cancellations Removal, One-Hot Basket Matrix |
| **Marwa** | Association Mining Engineer | Apriori vs. FP-Growth, Rule Mining (Support, Confidence, Lift), Recommender Engine |

---

## 3. Project Architecture
```text
Data-Mining-Project/
├── datasets/
│   ├── classification/      # Raw and processed Telco Churn datasets
│   ├── regression/          # Raw and processed Appliances Energy datasets
│   └── association/         # Raw and processed Online Retail datasets
├── notebooks/
│   ├── 01_classification_eda.ipynb
│   ├── 02_classification_models.ipynb
│   ├── 03_regression_eda.ipynb
│   ├── 04_regression_models.ipynb
│   ├── 05_association_eda.ipynb
│   └── 06_association_rules.ipynb
├── src/
│   ├── contracts.py         # Inter-module integration contracts & validation
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Preprocessing pipeline & transformers
│   │   └── model.py         # Model training, evaluation & prediction
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Feature engineering & chronological splitting
│   │   └── model.py         # Regression models & residual analysis
│   ├── association/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Basket matrix generation
│   │   ├── model.py         # Frequent itemset mining & rules
│   │   └── recommendations.py # Real-time recommendation engine
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py       # Shared evaluation metric helpers
│       └── visualization.py # ACS/Publication-ready plotting routines
├── models/
│   ├── classification/      # churn_pipeline.pkl
│   ├── regression/          # energy_pipeline.pkl
│   └── association/         # association_rules.csv
├── results/
│   ├── classification/      # Metrics, confusion matrix, ROC curves
│   ├── regression/          # Metrics, actual vs predicted, residual plots
│   └── association/         # Comparison tables, rules scatter plots
├── app/
│   ├── app.py               # Main Streamlit entrance
│   └── pages/
│       ├── 1_Classification_Churn.py
│       ├── 2_Regression_Energy.py
│       └── 3_Association_Recommendation.py
├── docs/
│   ├── contracts.md         # API & interface specifications
│   └── architecture.md      # Detailed system architecture
├── tests/
│   ├── mocks/               # Lightweight mock datasets for testing
│   └── test_contracts.py    # Integration contract verification
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Git Workflow & Governance
- **`main`**: Production-ready code only. Tagged releases for submission.
- **`develop`**: Central integration branch. All feature branches merge here via reviewed Pull Requests (PRs).
- **Feature Branches**:
  - `feature/classification-preprocessing` (Febronia)
  - `feature/classification-modeling` (Rovan)
  - `feature/regression-preprocessing` (Monzer)
  - `feature/regression-modeling` (Ebrahim)
  - `feature/association-preprocessing` (Basmala)
  - `feature/association-modeling` (Marwa)
  - `feature/streamlit-dashboard` (Mostafa)

---

## 5. Quick Start & Execution

### Installation
```bash
git clone <repository-url>
cd DM-Project
pip install -r requirements.txt
```

### Validate Contracts
```bash
python -c "from src.contracts import validate_contracts; validate_contracts()"
```

### Run Streamlit Interactive Application
```bash
streamlit run app/app.py
```
