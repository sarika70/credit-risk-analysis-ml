# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:38:42 2025

@author: mjayant
"""

import shap
import pandas as pd
import numpy as np

def get_shap_explanation(pipeline, input_df):
    """
    Generates dynamic 'Reason Codes' for credit decisions using SHAP.
    Identifies which features pushed the probability of default up or down.
    """
    try:
        # 1. Access the preprocessing and model stages
        preprocessor = pipeline.named_steps['preprocessing']
        stacking_model = pipeline.named_steps['model']
        
        # 2. Extract feature names after one-hot encoding
        # This is critical to map SHAP values back to readable names
        feature_names = preprocessor.get_feature_names_out()
        
        # 3. Transform input and select a base model for explanation
        # Stacking models are complex; we explain the strongest base learner (e.g., Random Forest)
        # to provide reliable intuition for the final decision.
        X_transformed = preprocessor.transform(input_df)
        base_rf_model = stacking_model.estimators_[0] 
        
        # 4. Calculate SHAP values
        explainer = shap.TreeExplainer(base_rf_model)
        # For classifiers, shap_values is often a list [prob_0, prob_1]
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle different SHAP output formats (TreeExplainer vs Linear)
        if isinstance(shap_values, list):
            # Focus on the 'Default' class (usually index 1)
            vals = shap_values[1][0] # array of features are retrieved here.
        else:
            vals = shap_values[0]

        # 5. Create a mapping of Feature -> Impact
        feature_impacts = pd.Series(vals, index=feature_names)
        
        # 6. Extract Top 3 Positive Influencers (Increased Risk) 
        # and Top 3 Negative Influencers (Decreased Risk)
        top_positive = feature_impacts.sort_values(ascending=False).head(3)
        top_negative = feature_impacts.sort_values(ascending=True).head(3)
        
        reasons = []
        
        if any(top_positive > 0):
            pos_features = [f.replace('num__', '').replace('cat__', '') for f in top_positive.index if top_positive[f] > 0]
            reasons.append(f"Risk Factors: {', '.join(pos_features)}")
            
        if any(top_negative < 0):
            neg_features = [f.replace('num__', '').replace('cat__', '') for f in top_negative.index if top_negative[f] < 0]
            reasons.append(f"Mitigating Factors: {', '.join(neg_features)}")

        return " | ".join(reasons) if reasons else "No dominant risk drivers identified."

    except Exception as e:
        # Log the specific error for debugging in production
        print(f"SHAP Error: {str(e)}")
        return "Reason codes unavailable due to calculation complexity."