from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE, 'model', 'best_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'model', 'scaler.pkl'))
with open(os.path.join(BASE, 'model', 'meta.json')) as f:
    meta = json.load(f)
meta['results'] = dict(list(meta['results'].items())[:-3])


FEATURES = meta['features']

# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        import pandas as pd
        values = {feat: float(data.get(feat, 0)) for feat in FEATURES}
        arr = pd.DataFrame([values])
        arr_sc = scaler.transform(arr)
        prob = model.predict_proba(arr_sc)[0][1]
        pred = int(prob >= 0.5)

        # Risk factors
        factors = []
        if data.get('OverTime', 0) == 1:           factors.append('Working overtime')
        if float(data.get('MonthlyIncome', 0)) < 3000: factors.append('Low monthly income')
        if float(data.get('JobSatisfaction', 4)) <= 2:  factors.append('Low job satisfaction')
        if float(data.get('WorkLifeBalance', 4)) <= 2:  factors.append('Poor work-life balance')
        if float(data.get('DistanceFromHome', 0)) > 20: factors.append('Long commute')
        if float(data.get('NumCompaniesWorked', 0)) > 5: factors.append('High job-hopping history')
        if float(data.get('YearsSinceLastPromotion', 0)) > 10: factors.append('No recent promotion')
        if float(data.get('EnvironmentSatisfaction', 4)) <= 2: factors.append('Low environment satisfaction')

        return jsonify({
            'probability': round(float(prob) * 100, 1),
            'prediction': pred,
            'risk_level': 'High' if prob > 0.65 else ('Medium' if prob > 0.35 else 'Low'),
            'risk_factors': factors,
            'best_model': meta['best_model'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-stats')
def model_stats():
    return jsonify(meta)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
