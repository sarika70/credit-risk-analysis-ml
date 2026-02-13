# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:34:59 2025

@author: mjayant
"""

import sys
import os
from models.train_ensemble import train_model


def setup():
    """Create necessary directory structure for Windows."""
    folders = [
        'data/raw', 'data/processed', 'artifacts', 
        'monitoring', 'explainability', 'models', 
        'features', 'ui', 'services'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print(" === System Directories Ready.==== ")

if __name__ == "__main__":
    setup()
    print("-" * 50)
    print("STARTING: Credit Risk Estimation Engine Pipeline")
    print("-" * 50)
    train_model()
    print("-" * 50)
    print("SUCCESS: Model ready for downstream integration.")