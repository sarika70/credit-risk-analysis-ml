# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:38:12 2025

@author: mjayant
"""

import numpy as np

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index."""
    def get_dist(data, bins):
        return np.histogram(data, bins=bins)[0] / len(data)
    
    breakpoints = np.percentile(expected, np.arange(0, 100, 100/buckets))
    e_dist = get_dist(expected, breakpoints)
    a_dist = get_dist(actual, breakpoints)
    
    a_dist = np.where(a_dist == 0, 0.0001, a_dist)
    return np.sum((e_dist - a_dist) * np.log(e_dist / a_dist))


