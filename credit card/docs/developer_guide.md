# Developer Integration & Architecture Guide

This developer guide provides an architectural overview, schema definitions, and REST integration standards for the Aegis Credit Card Approval Prediction System.

---

## 1. System Architecture

The project is structured following the MVC (Model-View-Controller) architecture:
- **Model**: Python machine learning pipeline components (`preprocessing.py`, `feature_engineering.py`, `model.py`, `predict.py`).
- **View**: Jinja2 HTML templates, CSS animations, and Chart.js dashboards.
- **Controller**: Flask routing controllers (`app.py`), database queries (`database.py`), and validation parameters.

```
                  ┌──────────────┐
                  │  Web Browser │
                  └──────┬───────┘
                         │ HTTP Requests
                         ▼
                  ┌──────────────┐
                  │ Flask Controller│
                  │   (app.py)   │
                  └──────┬───────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  ┌──────────────┐                ┌──────────────┐
  │  ML Engine   │                │ SQLite DB    │
  │ (predict.py) │                │(database.py) │
  └──────┬───────┘                └──────────────┘
         │
         ▼
  ┌──────────────┐
  │ Serialized   │
  │ Model (.pkl) │
  └──────────────┘
```

---

## 2. Preprocessing & Feature Engineering Pipeline

### A. Missing Value Imputation
- **Numeric features**: Imputed using the column `median` value to prevent outlier skew.
- **Categorical features**: Imputed using the column `most_frequent` (mode) category.

### B. Outlier Capping
- Numeric features are bounded using standard Interquartile Range boundaries:
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
- Outliers are clipped/capped to boundaries rather than discarded, preserving sample volume.

### C. Feature Engineering (Domain Ratios)
- **Debt-to-Income Ratio (DTI)**: $\text{DTI} = \frac{\text{Loan Amount} / 60}{\text{Monthly Income}}$
- **Leverage Coefficient**: $\text{Leverage} = \frac{(\text{Existing Loans} + 1) \times \text{Loan Amount}}{\text{Annual Income}}$
- **Income Consistency**: $\text{Consistency} = \frac{\text{Annual Income}}{\text{Monthly Income} \times 12}$

---

## 3. Database Schema

We use SQLite for audit persistence. The `predictions` schema is defined below:

| Column Name | Data Type | Key / Constraint | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Auto-incremented identifier |
| `timestamp` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Date and time of audit log |
| `gender` | `TEXT` | | Demographics |
| `age` | `REAL` | | Demographics |
| `income` | `REAL` | | Demographics |
| `employment_type` | `TEXT` | | Demographics |
| `employment_years`| `REAL` | | Demographics |
| `education` | `TEXT` | | Demographics |
| `marital_status` | `TEXT` | | Demographics |
| `dependents` | `INTEGER` | | Demographics |
| `occupation` | `TEXT` | | Demographics |
| `residence_type` | `TEXT` | | Demographics |
| `credit_history` | `TEXT` | | Demographics |
| `existing_loans` | `INTEGER` | | Demographics |
| `payment_history` | `TEXT` | | Demographics |
| `annual_income` | `REAL` | | Demographics |
| `loan_amount` | `REAL` | | Demographics |
| `loan_purpose` | `TEXT` | | Demographics |
| `prediction` | `TEXT` | | 'Approved' or 'Rejected' |
| `confidence` | `REAL` | | Prediction class confidence |
| `probability` | `REAL` | | Raw probability value |
| `risk_level` | `TEXT` | | Low, Medium, or High Risk |
| `approval_score` | `REAL` | | Computed score (300 to 850) |

---

## 4. API Endpoint Integration

### Post Inference
- **Endpoint**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Example Python Request:
```python
import requests

payload = {
    "gender": "Female",
    "age": 29,
    "income": 6500,
    "employment_type": "Salaried",
    "employment_years": 5.0,
    "education": "Post-Graduate",
    "marital_status": "Single",
    "dependents": 0,
    "occupation": "Engineer",
    "residence_type": "Owned",
    "credit_history": "Good",
    "existing_loans": 0,
    "payment_history": "No Delays",
    "annual_income": 78000,
    "loan_amount": 20000,
    "loan_purpose": "Home"
}

response = requests.post("http://localhost:5000/api/predict", json=payload)
print(response.json())
```
