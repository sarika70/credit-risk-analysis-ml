# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:39:16 2025

@author: mjayant
"""

import yaml
import os
import math

def load_config():
    """Load thresholds from config.yaml for centralized governance."""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return None

def probability_to_score(prob, factor=50, offset=500):
    # Prevent log(0) errors
    prob = max(min(prob, 0.999), 0.001)
    odds = (1 - prob) / prob
    score = offset + factor * math.log(odds)
    return int(max(min(score, 850), 300))

def get_realtime_risk_details(prob):
    """
    Refined logic to ensure 'Borderline' cases (like score 550) 
    are referred to manual review.
    """
    credit_score = probability_to_score(prob)
    
    # 1. High Risk / Auto-Decline (PD > 70% or Score < 450)
    if prob > 0.70 or credit_score < 450:
        return {
            "risk_band": "High Risk",
            "credit_score": credit_score,
            "decision": "DECLINE",
            "action_code": "D01",
            "color": "#ef4444" 
        }
    
    # 2. Medium Risk / Manual Review (PD > 25% or Score < 620)
    # This captures your 550 score / 26% PD case perfectly.
    elif prob > 0.25 or credit_score < 620:
        return {
            "risk_band": "Medium Risk",
            "credit_score": credit_score,
            "decision": "REFER TO UNDERWRITER",
            "action_code": "R05",
            "color": "#f97316" # Orange
        }
    
    # 3. Low Risk / Auto-Approve
    else:
        return {
            "risk_band": "Low Risk",
            "credit_score": credit_score,
            "decision": "AUTO-APPROVE",
            "action_code": "A00",
            "color": "#22c55e" # Green
        }

# Legacy support for older calls
def get_risk_band(prob):
    details = get_realtime_risk_details(prob)
    return details["risk_band"]