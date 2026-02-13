# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:34:31 2025

@author: mjayant
"""

import os
import sys
from models.train_boosted import train_boosted_ensemble
from models.train_voting import train_voting_ensemble
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
    
    #  Run the Boosted Ensemble by default (Highest Performance)
    print("\n--- Training Production  (XGBoost) ---")
    train_boosted_ensemble()
    
    # You can also run others as needed:
    # print("\n--- Training Production  (Voting Ensemble) ---")    
    # train_voting_ensemble()
    # print("\n--- Training Production  (Logistics) ---")
    # train_logistic_baseline()
    
    print("\n System ready. Start the API with: python app.py")
