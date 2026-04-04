"""
Statistical utilities for PROMETHEUS-EBM Research-Grade validation.
"""
import random
import math
import numpy as np
from typing import List, Tuple


def permutation_pvalue(a_vals: List[float], b_vals: List[float], 
                       rounds: int = 3000, seed: int = 42) -> float:
    """
    Computes a two-sided p-value using a permutation test.
    Used to determine if the difference between two models is statistically significant.
    """
    a = [float(v) for v in a_vals if v is not None]
    b = [float(v) for v in b_vals if v is not None]
    
    if not a or not b:
        return float('nan')
        
    observed_diff = abs(np.mean(a) - np.mean(b))
    combined = a + b
    n_a = len(a)
    
    rng = random.Random(seed)
    extreme_count = 0
    
    for _ in range(rounds):
        rng.shuffle(combined)
        a_star = combined[:n_a]
        b_star = combined[n_a:]
        delta = abs(np.mean(a_star) - np.mean(b_star))
        if delta >= observed_diff:
            extreme_count += 1
            
    return (extreme_count + 1.0) / (rounds + 1.0)


def cohens_h(p1: float, p2: float) -> float:
    """
    Computes Cohen's h for the difference between two proportions.
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    
    Interpretation:
    - 0.2: Small
    - 0.5: Medium
    - 0.8: Large
    """
    p1 = min(max(float(p1), 1e-9), 1.0 - 1e-9)
    p2 = min(max(float(p2), 1e-9), 1.0 - 1e-9)
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def effect_label(h: float) -> str:
    """Returns a qualitative label for Cohen's h."""
    ah = abs(h)
    if ah < 0.2: return 'negligible'
    if ah < 0.5: return 'small'
    if ah < 0.8: return 'medium'
    return 'large'
