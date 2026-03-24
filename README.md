# Employee Attrition Predictor
### CS33002 — Applications Development Laboratory

A Flask-based ML web application that predicts whether an employee is likely to leave
an organisation, using the IBM HR Analytics dataset.

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model/ artifacts)
python train_model.py

# 3. Start the Flask app
python app.py

# 4. Open in browser
http://localhost:5000
```

---

## Project Structure

```
attrition_app/
├── app.py              ← Flask backend (routes, prediction API)
├── train_model.py      ← Model training + comparison script
├── requirements.txt    ← Python dependencies
├── model/
│   ├── best_model.pkl  ← Trained best model (Logistic Regression)
│   ├── scaler.pkl      ← StandardScaler for feature normalisation
│   └── meta.json       ← Model comparison results + metadata
└── templates/
    └── index.html      ← Frontend (HTML/CSS/JS)
```

---

## Features

- **5 ML models compared**: Logistic Regression, Random Forest, Decision Tree, K-NN, SVM
- **Best model auto-selected** by F1-score
- **REST API** at `/predict` (POST, JSON) — returns probability, risk level, risk signals
- **Model stats** at `/model-stats` (GET)
- **Frontend**: real-time prediction, model comparison table, risk factor tags

---

## Input Parameters

| Feature | Description |
|---|---|
| Age | Employee age (18–60) |
| Monthly Income | Salary in $/₹ |
| Job Satisfaction | 1 (Low) – 4 (High) |
| Work-Life Balance | 1 (Low) – 4 (High) |
| Environment Satisfaction | 1 (Low) – 4 (High) |
| Overtime | Yes / No |
| Distance from Home | km (1–29) |
| Years at Company | 0–40 |
| Num Companies Worked | 0–9 |
| Years Since Last Promotion | 0–15 |
| Education | 1–5 |
| Performance Rating | 1–4 |

---

## API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 28,
    "MonthlyIncome": 2500,
    "JobSatisfaction": 2,
    "WorkLifeBalance": 2,
    "OverTime": 1,
    "NumCompaniesWorked": 4,
    "DistanceFromHome": 22,
    "YearsAtCompany": 1,
    "YearsSinceLastPromotion": 0,
    "EnvironmentSatisfaction": 2,
    "Education": 3,
    "PerformanceRating": 3
  }'
```

**Response:**
```json
{
  "probability": 74.3,
  "prediction": 1,
  "risk_level": "High",
  "risk_factors": ["Working overtime", "Low monthly income", "Low job satisfaction"],
  "best_model": "Logistic Regression"
}
```

---

## Dataset
- **Source**: IBM HR Analytics Employee Attrition Dataset (Kaggle)
- **Records**: 1470 employees
- **Target**: Attrition (Yes/No) — ~13.7% positive class
