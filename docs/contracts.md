# Integration Contracts & Data Exchange Interfaces

**Project:** Data Mining 3-Task Production System (7-Day Sprint)  
**Author:** Mostafa (Team Leader & Integration Engineer)

---

## Purpose
This document defines the binding contracts between data engineers, modeling engineers, and the Streamlit dashboard integration. Following these contracts guarantees that work proceeds in parallel without blocking any team member.

---

## 1. Classification Contract: Customer Churn
**Engineers:** Febronia (Data/Preprocessing) & Rovan (Modeling)

### A. Preprocessing Output Contract (Febronia -> Rovan)
- **Function Signature:**
  ```python
  def prepare_churn_data(filepath: str, apply_smote: bool = True) -> tuple:
      """
      Returns:
          X_train_proc (np.ndarray): Scaled, encoded training features (SMOTE applied if True).
          X_test_proc (np.ndarray): Scaled, encoded test features (Unseen, NO SMOTE).
          y_train (pd.Series/np.ndarray): Binary labels (0 or 1).
          y_test (pd.Series/np.ndarray): Binary labels (0 or 1).
          preprocessor (ColumnTransformer): Fitted preprocessor pipeline.
      """
  ```
- **Anti-Leakage Rule:** `SMOTE` and `fit_transform` must ONLY be applied to `X_train`. `X_test` receives only `transform`.

### B. Inference Contract (Rovan -> Mostafa / Streamlit)
- **File Artifact:** `models/classification/churn_pipeline.pkl`
- **Function Signature:**
  ```python
  def predict_churn(customer_data: dict) -> dict:
      """
      Input: Dict with 19 customer attributes.
      Output:
      {
          'prediction': 'Customer Likely to Churn' | 'Customer Likely to Stay',
          'churn_probability': 0.78,
          'churn_flag': 1,
          'model_used': 'Random Forest Classifier'
      }
      """
  ```

---

## 2. Regression Contract: Energy Consumption
**Engineers:** Monzer (Data/Preprocessing) & Ebrahim (Modeling)

### A. Preprocessing Output Contract (Monzer -> Ebrahim)
- **Function Signature:**
  ```python
  def prepare_energy_data(filepath: str, test_ratio: float = 0.2) -> tuple:
      """
      Returns:
          X_train_proc (np.ndarray): Scaled features for older 80% chronologically.
          X_test_proc (np.ndarray): Scaled features for newest 20% chronologically.
          y_train (pd.Series): Target 'Appliances' in Wh.
          y_test (pd.Series): Target 'Appliances' in Wh.
          preprocessor (Pipeline): Fitted StandardScaler pipeline.
          feature_names (list[str]): Names of processed features.
      """
  ```
- **Anti-Leakage Rule:** Temporal split must be strictly chronological (Index-based, NO random shuffle).

### B. Inference Contract (Ebrahim -> Mostafa / Streamlit)
- **File Artifact:** `models/regression/energy_pipeline.pkl`
- **Function Signature:**
  ```python
  def predict_energy(features: dict) -> dict:
      """
      Input: Dict of sensor readings and temporal values (hour, day_of_week).
      Output:
      {
          'predicted_energy_Wh': 85.4,
          'unit': 'Watt-hour (Wh)',
          'model_used': 'XGBoost Regressor'
      }
      """
  ```

---

## 3. Association Rules Contract: Product Recommendation
**Engineers:** Basmala (Data/Basket Matrix) & Marwa (Mining/Recommender Engine)

### A. Basket Matrix Contract (Basmala -> Marwa)
- **File Artifact:** `datasets/association/basket_matrix.pkl`
- **Contract:**
  - Structure: `pd.DataFrame`
  - Index: `InvoiceNo`
  - Columns: Unique Item `Description`
  - Values: Binary boolean (`True`/`False` or `1`/`0`)
  - Row constraint: Every transaction row must have `sum >= 2`.

### B. Recommendation Engine Contract (Marwa -> Mostafa / Streamlit)
- **File Artifact:** `models/association/association_rules.csv`
- **Function Signature:**
  ```python
  def recommend_products(selected_items: list[str], top_n: int = 5) -> list[dict]:
      """
      Input: List of selected item names.
      Output:
      [
          {
              'recommended_item': 'RED HANGING HEART T-LIGHT HOLDER',
              'confidence': 0.82,
              'lift': 4.5,
              'support': 0.045,
              'based_on_rule': 'WHITE HANGING HEART T-LIGHT HOLDER -> RED HANGING HEART T-LIGHT HOLDER'
          },
          ...
      ]
      """
  ```
