"""
Audit utilities for PROMETHEUS-EBM contamination detection.
"""
import re
from typing import List, Dict, Set, Any


def get_jaccard_similarity(str1: str, str2: str) -> float:
    """Computes Jaccard similarity between two strings."""
    def get_tokens(s: str) -> Set[str]:
        # Normalize and tokenize
        s = s.lower().strip()
        tokens = re.findall(r'\w+', s)
        return set(tokens)
    
    tokens1 = get_tokens(str1)
    tokens2 = get_tokens(str2)
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return float(len(intersection)) / len(union)


def audit_contamination(base_examples: List[Dict[str, Any]], 
                        probe_examples: List[Dict[str, Any]], 
                        threshold: float = 0.90) -> Dict[str, Any]:
    """
    Audits for data leakage between base evaluation set and probe sets.
    Checks for exact text overlap and near-duplicates using Jaccard similarity.
    """
    id_overlap = []
    text_overlap = []
    near_duplicates = []
    
    base_ids = {str(ex.get('problem_id')) for ex in base_examples}
    
    for p in probe_examples:
        p_id = str(p.get('problem_id'))
        p_text = str(p.get('question') or p.get('user', ''))
        
        # 1. ID Overlap
        if p_id in base_ids:
            id_overlap.append(p_id)
            
        # 2. Text Overlap / Near-duplicates
        for b in base_examples:
            b_text = str(b.get('question') or b.get('user', ''))
            
            if p_text == b_text:
                text_overlap.append((p_id, b.get('problem_id')))
            else:
                sim = get_jaccard_similarity(p_text, b_text)
                if sim >= threshold:
                    near_duplicates.append((p_id, b.get('problem_id'), sim))
                    
    return {
        'id_overlap_count': int(len(id_overlap)),
        'exact_text_overlap_count': int(len(text_overlap)),
        'near_duplicate_count_jaccard_ge_0_90': int(len(near_duplicates)),
        'id_overlap_list': id_overlap,
        'near_duplicates': near_duplicates,
        'pass': bool(not id_overlap and not text_overlap and not near_duplicates)
    }
