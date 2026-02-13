# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:49:47 2025

@author: mjayant
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5002/predict"

# Define Test Cases based on HMEQ feature logic
test_cases = [
    {
        "name": "Low Risk (Premier Borrower)",
        "description": "High value, long job stability, zero delinquencies, low debt-to-income.",
        "payload": {
            "LOAN": 10000,
            "MORTDUE": 40000,
            "VALUE": 120000,
            "YOJ": 15.0,
            "DEROG": 0,
            "DELINQ": 0,
            "CLAGE": 300.0,
            "NINQ": 0,
            "CLNO": 20,
            "DEBTINC": 20.5,
            "JOB": "ProfExe",
            "REASON": "HomeImp"
        }
    },
    {
        "name": "Medium Risk (Standard Borrower)",
        "description": "Average profile with some recent inquiries and moderate debt.",
        "payload": {
            "LOAN": 25000,
            "MORTDUE": 70000,
            "VALUE": 95000,
            "YOJ": 4.5,
            "DEROG": 0,
            "DELINQ": 1,
            "CLAGE": 120.0,
            "NINQ": 2,
            "CLNO": 15,
            "DEBTINC": 34.0,
            "JOB": "Office",
            "REASON": "DebtCon"
        }
    },
    {
        "name": "High Risk (Subprime Borrower)",
        "description": "Low tenure, high delinquencies, and critical debt-to-income ratio.",
        "payload": {
            "LOAN": 50000,
            "MORTDUE": 20000,
            "VALUE": 60000,
            "YOJ": 0.5,
            "DEROG": 3,
            "DELINQ": 4,
            "CLAGE": 40.0,
            "NINQ": 6,
            "CLNO": 10,
            "DEBTINC": 52.0,
            "JOB": "Sales",
            "REASON": "DebtCon"
        }
    }
]

def run_risk_tests():
    print(f"{'='*60}")
    print(f"RISK ENGINE SCENARIO TESTING")
    print(f"{'='*60}\n")

    for case in test_cases:
        print(f"Running: {case['name']}")
        print(f"Profile: {case['description']}")
        
        try:
            response = requests.post(BASE_URL, json=case['payload'])
            
            if response.status_code == 200:
                res = response.json()
                prob = res.get('probability_of_default', 0)
                score = res.get('credit_score', 'N/A')
                decision = res.get('decision', 'N/A')
                
                # Visual Indicator
                marker = "🟢" if "Approve" in decision else "🟡" if "Review" in decision else "🔴"
                
                print(f"Result: {marker} {decision}")
                print(f"Probability of Default: {prob:.4f}")
                print(f"Simulated Credit Score: {score}")
                
                # Dimensionality Check for the 52 vs 26 error
                shap = res.get('explanation', [])
                if isinstance(shap, list):
                    print(f"SHAP Values Length: {len(shap)} {'(⚠️ ERROR: 52!)' if len(shap) == 52 else '(✅ OK: 26)'}")
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    run_risk_tests()