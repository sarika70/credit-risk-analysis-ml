# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:38:22 2025

@author: mjayant
"""


def check_performance_degradation(current_ks, threshold=0.1):
    if current_ks < threshold:
        return "ALERT: Model Degradation"
    return "Stable"


