# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:57:15 2025

@author: mjayant
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import json
from datetime import datetime


class CreditRiskMonitor:
    """
    Module to calculate PSI (Population Stability Index), 
    K-S Statistics, and Drift metrics for Credit Risk.
    """
    def __init__(self, baseline_path="data/processed/training_reference.csv"):
        # Fix: Ensure path matches the actual training output directory
        if os.path.exists(baseline_path):
            self.baseline = pd.read_csv(baseline_path)
            # Ensure we have the probability column for scoring drift
            if 'predicted_prob' not in self.baseline and 'BAD' in self.baseline:
                # Fallback: if probs aren't in reference, use the target as a proxy for dist
                self.baseline['predicted_prob'] = self.baseline['BAD']
        else:
            print(f"Warning: Baseline file {baseline_path} missing. Using simulated distribution.")
            self.baseline = pd.DataFrame({
                'predicted_prob': np.random.beta(2, 5, 1000), 
                'DEBTINC': np.random.normal(30, 8, 1000)
            })

    def calculate_psi(self, expected, actual, buckets=10):
        """Calculates PSI between two populations with robust error handling."""
        try:
            # Drop NaNs to prevent calculation failure
            expected = expected.dropna()
            actual = actual.dropna()

            if len(actual) == 0:
                return 0.0

            # Use quantiles to define the buckets based on expected data
            breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
            
            # If expected data is too constant, breakpoints might not work
            if len(breakpoints) < 2:
                return 0.0

            def get_distribution(data, bins):
                hist, _ = np.histogram(data, bins=bins)
                return hist / len(data)

            e_percents = get_distribution(expected, breakpoints)
            a_percents = get_distribution(actual, breakpoints)
            
            # Clip to avoid log(0) and division by zero
            e_percents = np.clip(e_percents, 0.0001, None)
            a_percents = np.clip(a_percents, 0.0001, None)
            
            psi_val = np.sum((e_percents - a_percents) * np.log(e_percents / a_percents))
            return float(psi_val)
        except Exception as e:
            print(f"PSI Calculation Error: {e}")
            return 0.0

    def analyze_current_drift(self, log_path="logs/production_predictions.csv"):
        """Main entry point for the drift scheduler."""
        if not os.path.exists(log_path):
            return {"error": "No production data found. Run a few predictions in the dashboard first."}

        try:
            prod_df = pd.read_csv(log_path)
            
            if len(prod_df) < 5:
                return {"error": "Insufficient production data. Need at least 5 records for meaningful drift analysis."}

            # 1. Score Drift (PSI on Model Output)
            # Compares the 'predicted_prob' in training vs 'predicted_prob' in logs
            train_probs = self.baseline['predicted_prob']
            score_psi = self.calculate_psi(train_probs, prod_df['predicted_prob'])
            
            # 2. Feature Drift (PSI on Key Input: Debt-to-Income)
            # Uses DEBTINC as a representative feature for data drift
            train_debt = self.baseline['DEBTINC'] if 'DEBTINC' in self.baseline else pd.Series(np.random.normal(30, 8, 1000))
            debt_psi = self.calculate_psi(train_debt, prod_df['DEBTINC'])

            # 3. KS Test for Statistical Significance
            # Non-parametric test for distribution difference
            ks_stat, p_val = stats.ks_2samp(train_probs, prod_df['predicted_prob'])

            report = {
                "report_timestamp": datetime.now().isoformat(),
                "population_size": len(prod_df),
                "metrics": {
                    "score_drift_psi": round(score_psi, 4),
                    "feature_drift_psi": round(debt_psi, 4),
                    "ks_p_value": round(float(p_val), 4)
                },
                "status": {
                    "overall_health": "HEALTHY" if score_psi < 0.1 and p_val > 0.05 else "DRIFT_DETECTED",
                    "score_label": self._get_psi_label(score_psi),
                    "feature_label": self._get_psi_label(debt_psi)
                },
                "recommendation": "Maintain" if score_psi < 0.25 else "Retrain Model Immediately"
            }
            return report

        except Exception as e:
            return {"error": f"Internal analysis failure: {str(e)}"}

    def _get_psi_label(self, val):
        if val < 0.1: return "STABLE"
        if val < 0.25: return "MODERATE SHIFT"
        return "SIGNIFICANT DRIFT"

if __name__ == "__main__":
    monitor = CreditRiskMonitor()
    results = monitor.analyze_current_drift()
    print("\n--- Model Health Report ---")
    print(json.dumps(results, indent=4))