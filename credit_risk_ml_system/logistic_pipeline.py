# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:35:05 2025

@author: mjayant
"""


import os
import sys

from models.train_logistic import train_logistic_baseline

def setup():
    folders = [
        'data/raw', 'data/processed', 'artifacts', 
        'monitoring', 'explainability', 'models', 
        'features', 'ui', 'services', 'reports'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print(" Project Structure Initialized.")

if __name__ == "__main__":
    setup()
    
    # Example: Run the Boosted Ensemble by default (Highest Performance)
    print("\n--- Training Production Champion (XGBoost) ---")
    
    train_logistic_baseline()
    
    print("\n System ready. Start the API with: python app.py")