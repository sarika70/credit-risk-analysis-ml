# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:35:11 2025

@author: mjayant
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
from explainability.shap_explainer import get_shap_explanation
from services.scoring import get_realtime_risk_details

# Initialize Flask
app = Flask(__name__, static_folder='ui')
CORS(app) # Enable Cross-Origin Resource Sharing for the UI

MODEL_PATH = "artifacts/credit_risk_pipeline.pkl"

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
        
        
        # Server-side engineering to ensure columns are never "missing"
        data['COLLATERAL'] = data['VALUE'] - data['MORTDUE']
        data['L_P_RATIO'] = data['LOAN'] / data['VALUE'] if data['VALUE'] != 0 else 0
        data['L_C_RATIO'] = data['LOAN'] / data['COLLATERAL'] if data['COLLATERAL'] != 0 else 0
        data['C_P_RATIO'] = data['COLLATERAL'] / data['VALUE'] if data['VALUE'] != 0 else 0
        data['HIGH_DEBTINC_FLAG'] = 1 if data['DEBTINC'] > 45 else 0
        data['HAS_DEROG'] = 1 if data['DEROG'] > 0 else 0
        
        
        
        input_df = pd.DataFrame([data])
        
        
        
        # 2. Probability of Default (PD)
        prob = pipeline.predict_proba(input_df)[0][1]
        
        # 3. Industry Scoring (Real-time details)
        risk_details = get_realtime_risk_details(prob)
        
        # 4. Explainability (SHAP)
        explanation = get_shap_explanation(pipeline, input_df)
        
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
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("  Warning: Model artifact not found. Run main.py first!")
    
    print(" Credit Risk API & UI running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)