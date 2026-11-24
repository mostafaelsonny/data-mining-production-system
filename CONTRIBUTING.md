# Team Contribution Guidelines & Code Standards

Welcome to the **Data Mining 3-Task Production System** project repository!  
This guide outlines the mandatory software engineering standards, branching conventions, and review procedures for all 7 team members.

---

## 1. Branching Strategy
We enforce a strict Gitflow model:
- **`main`**: Production-ready code only. No direct commits allowed.
- **`develop`**: Central integration branch. All code is integrated here via Pull Requests.
- **`feature/*`**: Dedicated feature branches for individual assignments.

### Active Feature Branches:
| Member | Branch Name | Scope |
| :--- | :--- | :--- |
| **Febronia** | `feature/classification-preprocessing` | Churn EDA, Cleaning, SMOTE, Scaler |
| **Rovan** | `feature/classification-modeling` | Churn Classifiers, 6 Metrics, ROC Curve |
| **Monzer** | `feature/regression-preprocessing` | Energy EDA, Outliers, Chronological Split |
| **Ebrahim** | `feature/regression-modeling` | Energy Regressors, MSE/R², Residual Plot |
| **Basmala** | `feature/association-preprocessing` | Retail Cleaning, Basket Matrix |
| **Marwa** | `feature/association-modeling` | Apriori vs FP-Growth, Recommender Engine |
| **Mostafa** | `feature/streamlit-dashboard` | Integration & Multi-page Streamlit App |

---

## 2. Daily Workflow Step-by-Step

### A. Start of Day: Sync with `develop`
Before starting any new work, always pull the latest changes from `develop`:
```bash
git checkout develop
git pull origin develop
git checkout feature/your-feature-name
git merge develop
```

### B. Developing & Testing
- Keep your reusable Python functions inside `src/`.
- Use `notebooks/` for exploratory experiments, but export production functions to `src/`.
- Validate your code against the master contracts:
  ```bash
  python -c "from src.contracts import validate_contracts; validate_contracts()"
  ```

### C. Committing Changes
Write meaningful, conventional commit messages:
- `feat(...)`: New feature or implementation
- `fix(...)`: Bug fix or data cleanup correction
- `docs(...)`: Documentation or comments update
- `test(...)`: Unit tests addition

### D. Submitting a Pull Request (PR)
1. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request targeting `develop` (NEVER `main`).
3. Fill out the Pull Request template completely.
4. Notify **Mostafa (Project Leader)** on WhatsApp / Trello for review.
5. Address any review comments promptly. Once approved, Mostafa will merge the PR.

---

## 3. The 3 Golden Rules (Anti-Leakage & Reliability)
1. **Never fit on test data:** Any `fit` or `fit_transform` must happen exclusively on `X_train`.
2. **Never commit large raw datasets or secrets:** Datasets above 50MB should be managed via `.gitignore`.
3. **Always adhere to `src/contracts.py`:** Do not rename dictionary keys or alter function signatures without team-wide alignment.
