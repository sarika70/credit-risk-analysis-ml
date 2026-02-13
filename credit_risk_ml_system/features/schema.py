# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:37:04 2025

@author: mjayant
"""

import pandas as pd 

EXPECTED_SCHEMA = {
    "LOAN": "float64",
    "MORTDUE": "float64",
    "VALUE": "float64",
    "REASON": "object",
    "JOB": "object",
    "YOJ": "float64",
    "DEROG": "float64",
    "DELINQ": "float64",
    "CLAGE": "float64",
    "NINQ": "float64",
    "CLNO": "float64",
    "DEBTINC": "float64"
}

def validate_schema(df):
    """
    Check for required columns and verify data types for the HMEQ dataset.
    """
    for col, expected_type in EXPECTED_SCHEMA.items():
        # Check if column exists
        if col not in df.columns:
            print(f"Schema Error: Missing column {col}")
            return False
        
        # Verify data type (handles both standard and nullable pandas types)
        actual_type = df[col].dtype
        if not (pd.api.types.is_numeric_dtype(actual_type) if expected_type == "float64" else pd.api.types.is_object_dtype(actual_type)):
             # We allow some flexibility between int/float, but object must be object
             if expected_type == "object" and not pd.api.types.is_object_dtype(actual_type):
                print(f"Schema Error: Column {col} expected {expected_type}, got {actual_type}")
                return False
                
    return True