"""
Judge sensitivity utilities for PROMETHEUS-EBM.
"""
import random
from typing import List, Dict, Any, Optional


def measure_judge_sensitivity(items: List[Dict[str, Any]], 
                             judge_disagreement_rate: float,
                             sample_size: int = 180) -> Dict[str, Any]:
    """
    Simulates or measures the disagreement rate between independent judges.
    In a real scenario, this would involve re-running a sample with a different LLM judge.
    """
    total_sampled = min(len(items), sample_size)
    if total_sampled == 0:
        return {'pass': False, 'disagreement_rate': 1.0, 'reason': 'No items to sample'}
        
    disagreements = int(total_sampled * judge_disagreement_rate)
    disagreement_rate = float(disagreements) / total_sampled
    
    # 0.15 is the standard research-grade threshold for judge disagreement
    is_pass = disagreement_rate < 0.15
    
    return {
        'total_sampled': int(total_sampled),
        'disagreements': int(disagreements),
        'disagreement_rate': disagreement_rate,
        'threshold': 0.15,
        'pass': bool(is_pass)
    }
