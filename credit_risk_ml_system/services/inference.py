# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:39:06 2025

@author: mjayant
"""

import joblib

def load_latest_model(path="artifacts/credit_risk_pipeline.pkl"):
    return joblib.load(path)