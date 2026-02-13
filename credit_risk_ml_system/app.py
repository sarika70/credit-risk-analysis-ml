# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:49:09 2025

@author: mjayant
"""

# -*- coding: utf-8 -*-
"""
Integrated Credit Risk API with Monitoring Logging
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS # Cross Origin Resource Sharing 
import joblib
import pandas as pd
import numpy as np
import os
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


from explainability.shap_explainer import get_shap_explanation
from services.scoring import get_realtime_risk_details
from monitoring.drift_analysis import CreditRiskMonitor

# Initialize Flask
app = Flask(__name__, static_folder='ui')
CORS(app) 

# Configuration
MODEL_PATH = "artifacts/credit_risk_pipeline.pkl"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "production_predictions.csv")
# Constants
MODEL_PATH = "artifacts/credit_risk_pipeline.pkl"
LOG_FILE = "logs/production_predictions.csv"
BASELINE_FILE = "data/processed/training_reference.csv"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Ensure directories exist for logging
os.makedirs(LOG_DIR, exist_ok=True)

@app.route('/')
def index():
    """Serve the UI."""
    return send_from_directory('ui', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Load pipeline
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model artifact missing. Train a model first."}), 500
            
        pipeline = joblib.load(MODEL_PATH)
        data = request.get_json()
        print("Data coming from the UI - " , data)
        
        # --- FEATURE ENGINEERING (Required for the model) ---
        # We calculate these server-side so the UI/Bruno doesn't have to
        data['COLLATERAL'] = data.get('VALUE', 0) - data.get('MORTDUE', 0)
        data['L_P_RATIO'] = data.get('LOAN', 0) / data.get('VALUE', 1) if data.get('VALUE', 0) != 0 else 0
        data['L_C_RATIO'] = data.get('LOAN', 0) / data['COLLATERAL'] if data['COLLATERAL'] != 0 else 0
        data['C_P_RATIO'] = data['COLLATERAL'] / data.get('VALUE', 1) if data.get('VALUE', 0) != 0 else 0
        data['HIGH_DEBTINC_FLAG'] = 1 if data.get('DEBTINC', 0) > 45 else 0
        data['HAS_DEROG'] = 1 if data.get('DEROG', 0) > 0 else 0
        
        # Convert to DataFrame for the Scikit-Learn Pipeline
        input_df = pd.DataFrame([data])
        
        # 2. Probability of Default (PD)
        prob = pipeline.predict_proba(input_df)[0][1]
        
        # 3. Industry Scoring (Decisioning & Risk Bands)
        risk_details = get_realtime_risk_details(prob)
        
        # 4. Explainability (SHAP Reason Codes)
        explanation = get_shap_explanation(pipeline, input_df)
        
        # --- 5. DATA LOGGING FOR DRIFT MONITORING ---
        # We log everything: inputs, engineered features, and the prediction result
        log_data = data.copy()
        log_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data['predicted_prob'] = round(float(prob), 4)
        log_data['decision'] = risk_details['decision']
        
        log_df = pd.DataFrame([log_data])
        # Append to CSV; if file doesn't exist, write headers
        log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
        
        return jsonify({
            "probability_of_default": round(float(prob), 4),
            "credit_score": risk_details['credit_score'],
            "risk_band": risk_details['risk_band'],
            "decision": risk_details['decision'],
            "action_code": risk_details['action_code'],
            "explanation": explanation,
            "theme_color": risk_details['color']
        })
    
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
@app.route('/api/drift-report')
def drift_report():
    try:
        monitor = CreditRiskMonitor(baseline_path=BASELINE_FILE)
        report = monitor.analyze_current_drift(log_path=LOG_FILE)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/eda-report')
def eda_report():
    """Generates the EDA plot as a base64 string for the UI."""
    try:
        df = pd.read_csv("data/raw/hmeq.csv")
        df_calc = df.copy()
        df_calc['L_P_RATIO'] = df_calc['LOAN'] / df_calc['VALUE'].replace(0, np.nan)
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.countplot(x='BAD', data=df, ax=axes[0,0], palette='viridis')
        axes[0,0].set_title('Target Distribution (BAD)')
        
        sns.histplot(x='DEBTINC', hue='BAD', data=df, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Debt-to-Income Impact')
        
        sns.boxplot(x='BAD', y='L_P_RATIO', data=df_calc, ax=axes[1,0])
        axes[1,0].set_ylim(0, 1.5)
        axes[1,0].set_title('LTV Ratio by Outcome')
        
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr[['BAD']].sort_values(by='BAD'), annot=True, cmap='RdYlGn', ax=axes[1,1])
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({"image": plot_url, "summary": "EDA generated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(" [!] WARNING: Model artifact not found at " + MODEL_PATH)
    
    print(f" [*] Credit Risk Engine active: http://127.0.0.1:5000")
    print(f" [*] Real-time monitoring enabled. Logs: {LOG_FILE}")
    app.run(debug=True, port=5002)