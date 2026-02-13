# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:36:53 2025

@author: mjayant
"""

import pandas as pd
import numpy as np


from sklearn.impute import SimpleImputer

def create_features(df):
    """
    Advanced Feature Engineering for HMEQ Credit Risk.
    Includes Ratio Analysis, Outlier Clipping, and Robust Imputation.
    """
    df = df.copy() # deep copy
    
    # 1. Standardize Target
    if 'BAD' in df.columns:
        df = df.rename(columns={'BAD': 'target'})
    
    y = df["target"]
    X = df.drop(columns=["target"])

    # 2. Advanced Ratio Engineering (Requested Features)
    # Handle zeros in denominators to avoid infinity
    X['VALUE'] = X['VALUE'].replace(0, np.nan)
    X['MORTDUE'] = X['MORTDUE'].replace(0, np.nan)

    # Collateral: The equity buffer in the home
    X['COLLATERAL'] = X['VALUE'] - X['MORTDUE']
    
    # Loan-to-Property (LTV): Core industry risk metric
    X['L_P_RATIO'] = X['LOAN'] / X['VALUE']
    
    # Loan-to-Collateral: Risk relative to available equity
    # We clip this because collateral can be near zero or negative
    X['L_C_RATIO'] = (X['LOAN'] / X['COLLATERAL']).clip(-10, 10)
    
    # Collateral-to-Property: 'Skin in the game' ratio
    X['C_P_RATIO'] = X['COLLATERAL'] / X['VALUE']

    # 3. Handling Numerical Missing Values & Outliers
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    
    # Strategy: Median Imputation is more robust than mean for HMEQ
    for col in num_cols:
        # Fill missing values with median
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        
        # Outlier Clipping (99th percentile) to prevent extreme values from biasing the model
        upper_limit = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=upper_limit)

    # 4. Handling Categorical Features (REASON, JOB)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        # Cast to category for memory efficiency
        X[col] = X[col].astype('category')
        
        # Add a specific "Unknown" category for missing values
        # This allows the model to learn if "Missing" itself is a risk signal
        if "Unknown" not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories("Unknown")
        X[col] = X[col].fillna("Unknown")

    # 5. Financial Indicator Binning (Optional but helpful)
    # Create a flag for people with high debt-to-income
    X['HIGH_DEBTINC_FLAG'] = (X['DEBTINC'] > 45).astype(int)
    
    # Create a flag for derogatory history
    X['HAS_DEROG'] = (X['DEROG'] > 0).astype(int)

    print(f"✅ Feature Engineering Complete. Engineered {X.shape[1]} predictors.")
    return X, y