import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ── Synthetic IBM-style dataset (1470 rows) ─────────────────────────────────
np.random.seed(42)
n = 1470

age           = np.random.randint(18, 61, n)
income        = np.random.randint(1009, 20000, n)
job_sat       = np.random.randint(1, 5, n)
wlb           = np.random.randint(1, 5, n)
overtime_raw  = np.random.choice(['Yes', 'No'], n, p=[0.28, 0.72])
overtime      = (overtime_raw == 'Yes').astype(int)
num_companies = np.random.randint(0, 10, n)
distance      = np.random.randint(1, 30, n)
tenure        = np.random.randint(0, 41, n)
yrs_promotion = np.random.randint(0, 16, n)
env_sat       = np.random.randint(1, 5, n)
education     = np.random.randint(1, 6, n)
perf_rating   = np.random.randint(1, 5, n)

# Logistic attrition model (mimics real IBM dataset patterns)
log_odds = (
    0.2
    + overtime      * 1.8
    - (income / 20000) * 2.5
    - (job_sat / 4) * 1.5
    - (wlb / 4)     * 1.2
    + (distance / 29) * 0.8
    + (num_companies / 9) * 1.0
    - (tenure / 40) * 1.2
    + (yrs_promotion / 15) * 0.6
    - (env_sat / 4) * 0.8
    - (age / 60)    * 0.5
    + np.random.normal(0, 0.8, n)
)
prob_attr = 1 / (1 + np.exp(-log_odds))
attrition = (np.random.rand(n) < prob_attr).astype(int)

df = pd.DataFrame({
    'Age': age,
    'MonthlyIncome': income,
    'JobSatisfaction': job_sat,
    'WorkLifeBalance': wlb,
    'OverTime': overtime,
    'NumCompaniesWorked': num_companies,
    'DistanceFromHome': distance,
    'YearsAtCompany': tenure,
    'YearsSinceLastPromotion': yrs_promotion,
    'EnvironmentSatisfaction': env_sat,
    'Education': education,
    'PerformanceRating': perf_rating,
    'Attrition': attrition
})

print(f"Dataset: {len(df)} rows | Attrition rate: {df['Attrition'].mean():.1%}")

# ── Preprocessing ─────────────────────────────────────────────────────────
FEATURES = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance',
            'OverTime', 'NumCompaniesWorked', 'DistanceFromHome',
            'YearsAtCompany', 'YearsSinceLastPromotion',
            'EnvironmentSatisfaction', 'Education', 'PerformanceRating']

X = df[FEATURES]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train multiple models ─────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'K-Nearest Neighbour': KNeighborsClassifier(n_neighbors=7),
    'Support Vector Machine': SVC(probability=True, random_state=42),
}

results = {}
best_f1, best_name, best_model = 0, None, None

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    preds = clf.predict(X_test_sc)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    cm   = confusion_matrix(y_test, preds).tolist()

    results[name] = {
        'accuracy':  round(acc  * 100, 2),
        'precision': round(prec * 100, 2),
        'recall':    round(rec  * 100, 2),
        'f1':        round(f1   * 100, 2),
        'confusion_matrix': cm,
    }
    print(f"  {name:<26} Acc={acc:.3f}  F1={f1:.3f}")

    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, clf

print(f"\nBest model: {best_name} (F1={best_f1:.3f})")

# ── Save artifacts ────────────────────────────────────────────────────────
joblib.dump(best_model, 'model/best_model.pkl')
joblib.dump(scaler,     'model/scaler.pkl')

meta = {
    'best_model':  best_name,
    'features':    FEATURES,
    'results':     results,
    'attrition_rate': round(df['Attrition'].mean() * 100, 1),
    'dataset_size': len(df),
}
with open('model/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Saved: model/best_model.pkl, model/scaler.pkl, model/meta.json")
