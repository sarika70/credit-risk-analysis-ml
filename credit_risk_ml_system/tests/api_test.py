# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:48:59 2025

@author: mjayant
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_health_check():
    print("Testing UI Availability...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

def test_prediction_and_shap():
    print("\nTesting Predict API (Risk Scoring & SHAP)...")
    payload = {
        "LOAN": 15000,
        "MORTDUE": 60000,
        "VALUE": 100000,
        "YOJ": 10,
        "DEROG": 0,
        "DELINQ": 0,
        "CLAGE": 250,
        "NINQ": 0,
        "CLNO": 25,
        "DEBTINC": 25.5,
        "JOB": "Office",
        "REASON": "DebtCon"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Prob: {data.get('probability_of_default')}")
        print(f"Decision: {data.get('decision')}")
        
        # Diagnostics for the 52 vs 26 error
        explanation = data.get('explanation', [])
        if isinstance(explanation, list):
            length = len(explanation)
            print(f"SHAP Values Length: {length}")
            if length == 52:
                print("❌ ERROR DETECTED: Returning 52 values (2 classes). Needs slicing [1].")
            elif length == 26:
                print("✅ SHAP Dimensions Correct (26).")
        else:
            print(f"Explanation Info: {explanation}")
    else:
        print(f"❌ Failed! Status: {response.status_code}")
        print(f"Error Detail: {response.text}")

def test_drift_report():
    print("\nTesting Drift Report API...")
    # This often returns 500 if the log file has mismatched columns
    response = requests.get(f"{BASE_URL}/api/drift-report")
    
    if response.status_code == 200:
        print("✅ Drift Report Generated Successfully")
        print(json.dumps(response.json(), indent=2)[:200] + "...")
    elif response.status_code == 404:
        print("⚠️  Status 404: Baseline or Log files not found. Run a few predictions first.")
    else:
        print(f"❌ Failed! Status: {response.status_code}")
        print("Debugging Tip: Check if 'timestamp' column in logs is being fed to drift calculation.")

def test_eda_report():
    print("\nTesting EDA Report API...")
    response = requests.get(f"{BASE_URL}/api/eda-report")
    if response.status_code == 200:
        print("✅ EDA Report Success (Image data received)")
    else:
        print(f"❌ EDA Failed! Status: {response.status_code}")

if __name__ == "__main__":
    if test_health_check():
        test_prediction_and_shap()
        # Give the file system a moment to write logs
        time.sleep(1)
        test_drift_report()
        test_eda_report()