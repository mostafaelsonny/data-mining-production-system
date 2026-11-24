## 📋 Pull Request Description

### 1. Related Task / Trello Card
- **Trello Card:** [Card Title or URL]
- **Assigned Member:** @[GitHub-Username]
- **Target Branch:** `develop`

---

### 2. Changes Introduced
- [ ] Classification (Task 4: Customer Churn)
- [ ] Regression (Task 4: Appliances Energy)
- [ ] Association Rules (Task 2: Product Recommendation)
- [ ] Dashboard / Streamlit
- [ ] Infrastructure / Utilities / Docs

**Detailed Description:**
<!-- Explain what was implemented or modified -->

---

### 3. Engineering & Anti-Leakage Checklist (Mandatory)
- [ ] **Contract Compliance:** All function signatures and returned dictionary keys strictly follow `src/contracts.py`.
- [ ] **Zero Data Leakage:** Preprocessing transformations (`fit`) and oversampling (`SMOTE`) are executed ONLY on `X_train`.
- [ ] **No Hardcoding:** Inference and evaluation use serialized pipelines or live inputs, never mock constants.
- [ ] **Test Coverage:** All unit and contract tests pass (`pytest tests/ -v`).
- [ ] **Clean Code:** Code is properly modularized inside `src/` (not just in `.ipynb`).

---

### 4. Verification Evidence
<!-- Attach command outputs, metrics tables, or screenshot links -->
```text
pytest tests/ -v output:
...
```
