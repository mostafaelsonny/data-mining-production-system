# Team Developer Setup Guide

**Welcome to the Project!**  
Follow this 5-minute setup guide to clone the repository and configure your development environment.

---

## 1. Prerequisites
- **Git** installed on your machine: [Download Git](https://git-scm.com/)
- **Python 3.10+ (Recommended Python 3.12)**: [Download Python](https://www.python.org/)
- **VS Code** (Recommended IDE) with Python extension installed.

---

## 2. Clone the Repository & Switch to Your Branch
Open your terminal (or Command Prompt / PowerShell):
```bash
git clone <repository-url>
cd DM-Project

# Check available branches
git branch -a

# Checkout the develop branch first
git checkout develop

# Switch to your assigned feature branch:
# Febronia:
git checkout feature/classification-preprocessing

# Rovan:
git checkout feature/classification-modeling

# Monzer:
git checkout feature/regression-preprocessing

# Ebrahim:
git checkout feature/regression-modeling

# Basmala:
git checkout feature/association-preprocessing

# Marwa:
git checkout feature/association-modeling

# Mostafa:
git checkout feature/streamlit-dashboard
```

---

## 3. Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

---

## 4. Install Project Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Verify Your Environment
Run the contract test suite to verify that your environment is 100% ready:
```bash
pytest tests/ -v
```
You should see:
```text
tests/test_contracts.py::test_full_contracts_self_check PASSED
tests/test_contracts.py::test_classification_mock_inference PASSED
tests/test_contracts.py::test_regression_mock_inference PASSED
tests/test_contracts.py::test_recommendation_mock_inference PASSED
```

---

## 6. Launch the Streamlit Dashboard (Optional)
```bash
streamlit run app/app.py
```
A new browser window will open at `http://localhost:8501`.
