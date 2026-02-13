# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:37:37 2025

@author: mjayant
"""

import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

def get_credit_metrics(y_true, y_probs):
    
    auc = roc_auc_score(y_true, y_probs)
    
    # Converts AUC to the Gini Coefficient (0 to 1 scale). 
    # In credit, Gini is the "common language." 
    # A Gini of 0.60 is generally considered very strong for a loan model.
    
    gini = 2 * auc - 1
    
    df = pd.DataFrame({'t': y_true, 'p': y_probs})
    
    #df[df.t==0].p: This selects the predicted probabilities for all people who did not default.
    #df[df.t==1].p: This selects the predicted probabilities for all people who did default.
    #stats.ks_2samp(...).statistic: It calculates the maximum vertical distance between the Cumulative Distribution Functions (CDFs) of these two groups
    
    # The statistic returned is the maximum gap between 
    # the two cumulative distributions.
    # Strong separation; the model clearly distinguishes 
    # between high and low risk.
    
    ks = stats.ks_2samp(df[df.t==0].p, df[df.t==1].p).statistic
    
    return {
        "AUC": float(auc),
        "Gini": float(gini),
        "KS_Statistic": float(ks)
    }