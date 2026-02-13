# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:58:21 2025

@author: mjayant
"""

import schedule
import time
from monitoring.drift_analysis import RiskMonitor
import joblib
import pandas as pd

def run_monthly_audit():
    print("  Starting Monthly Model Audit...")
    
    # 1. Load the model and the baseline training data
    pipeline = joblib.load("artifacts/credit_risk_pipeline.pkl")
    
    # 2. Fetch the LAST 30 DAYS of production data from your database/logs
    # For this example, we assume you log every 'predict' request to a CSV/DB
    production_logs = pd.read_csv("logs/production_predictions.csv")
    
    # 3. Initialize Monitor and Run Analysis
    monitor = RiskMonitor(baseline_path="data/training_reference.csv")
    report = monitor.generate_health_report(production_logs, pipeline)
    
    # 4. Action based on results
    if report["metrics"]["score_drift_psi"] > 0.25:
        send_alert_to_slack(f" CRITICAL DRIFT DETECTED: PSI={report['metrics']['score_drift_psi']}")
    else:
        print(" Model is stable.")

# Schedule to run at midnight on the 1st of every month
schedule.every().month.at("00:00").do(run_monthly_audit)

while True:
    schedule.run_pending()
    time.sleep(60)