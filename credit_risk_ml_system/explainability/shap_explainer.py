# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:46:01 2025

@author: mjayant
"""
import shap
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier

def get_shap_explanation(pipeline, input_df):
    """
    Generates dynamic 'Reason Codes' for credit decisions using SHAP.
    Optimized for Logistic Regression and Tree-based ensembles.
    """
    try:
        # 1. Access the preprocessing and model stages
        preprocessor = pipeline.named_steps['preprocessing']
        model = pipeline.named_steps['model']
        
        # 2. Extract feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        
        # 3. Transform input
        X_transformed = preprocessor.transform(input_df)
        
        # 4. Handle Model Type Dynamically
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
            # shap_values is usually (n_samples, n_features)
            vals = shap_values[0]

        elif hasattr(model, "estimators_"):
            # Stacking/Ensemble: Explain the first base learner
            base_learner = model.estimators_[0]
            explainer = shap.TreeExplainer(base_learner)
            shap_values = explainer.shap_values(X_transformed)
            
            # --- FIX FOR 52 vs 26 ERROR ---
            if isinstance(shap_values, list):
                # Binary class list [class_0, class_1] -> Select class 1
                vals = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                # Array with shape (samples, features, classes) -> Select class 1
                vals = shap_values[0, :, 1]
            else:
                # Standard (samples, features)
                vals = shap_values[0]
        
        else:
            # Standard Tree model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

        # 5. Final Dimensionality Safeguard
        # If vals still has 52 elements for 26 features, take the second half (class 1)
        if len(vals) == 2 * len(feature_names):
            vals = vals[len(feature_names):]
        # If it's still a 2D array, flatten it to 1D
        if len(vals.shape) > 1:
            vals = vals.flatten()
            
        # 6. Create a mapping of Feature -> Impact
        feature_impacts = pd.Series(vals, index=feature_names)
        
        # 7. Extract Top 3 Impacting Features
        top_positive = feature_impacts.sort_values(ascending=False).head(3)
        top_negative = feature_impacts.sort_values(ascending=True).head(3)
        
        reasons = []
        
        # Risk Factors (Positive SHAP values increase probability of default)
        pos_features = [f.split("__")[-1] for f in top_positive.index if top_positive[f] > 1e-9]
        if pos_features:
            reasons.append(f"Risk Factors: {', '.join(pos_features)}")
            
        # Mitigating Factors (Negative SHAP values decrease probability of default)
        neg_features = [f.split("__")[-1] for f in top_negative.index if top_negative[f] < -1e-9]
        if neg_features:
            reasons.append(f"Mitigating Factors: {', '.join(neg_features)}")

        return " | ".join(reasons) if reasons else "Broad risk distribution identified."

    except Exception as e:
        print(f"SHAP Explainer Error: {str(e)}")
        return "Reason codes unavailable for this model configuration."