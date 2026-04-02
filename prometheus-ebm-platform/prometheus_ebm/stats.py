"""
stats.py — Statistical utilities for PROMETHEUS-EBM

Bootstrap confidence intervals, Wilson score intervals,
Cohen's h effect sizes, and calibration error computation.
All statistical claims in our paper require these functions.
"""

import math
import random
from typing import List, Tuple, Optional


def bootstrap_ci(
    values: List[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a mean/proportion.

    Args:
        values: List of 0/1 values (for proportions) or continuous values
        n_boot: Number of bootstrap samples
        ci: Confidence level (default 95%)
        seed: Random seed for reproducibility

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))

    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot)]
    return sum(values) / n, lo, hi


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> Tuple[float, float, float]:
    """
    Wilson score interval for a proportion.
    Better than normal approximation for small samples.

    Args:
        successes: Number of successes
        total: Total trials
        z: Z-score for confidence level (1.96 for 95%)

    Returns:
        (proportion, ci_lower, ci_upper)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    spread = z * math.sqrt(
        (p * (1 - p) + z ** 2 / (4 * total)) / total
    ) / denom

    return p, max(0, centre - spread), min(1, centre + spread)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for comparing two proportions.

    Interpretation:
        |h| < 0.2  → negligible
        |h| < 0.5  → small
        |h| < 0.8  → medium
        |h| >= 0.8 → LARGE

    Args:
        p1, p2: Two proportions to compare

    Returns:
        Cohen's h value (positive means p1 > p2)
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def effect_magnitude(h: float) -> str:
    """Interpret Cohen's h magnitude."""
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def expected_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """
    Expected Calibration Error (ECE).
    Measures the gap between stated confidence and actual accuracy.

    A perfectly calibrated model has ECE = 0.
    Gemini 3.1 Pro: ECE = 0.347 (badly miscalibrated)
    Opus 4.6: ECE = 0.217 (best among tested models)

    Args:
        confidences: List of model confidence values [0, 1]
        correctness: List of boolean correctness flags
        n_bins: Number of calibration bins

    Returns:
        (ece_value, bin_details)
    """
    if not confidences:
        return 0.0, []

    n = len(confidences)
    bin_size = 1.0 / n_bins
    bins = []

    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size

        indices = [
            j for j in range(n)
            if lo <= confidences[j] < hi or (i == n_bins - 1 and confidences[j] == 1.0)
        ]

        if not indices:
            bins.append({
                'bin_lo': lo, 'bin_hi': hi,
                'count': 0, 'avg_confidence': 0, 'avg_accuracy': 0,
                'gap': 0,
            })
            continue

        avg_conf = sum(confidences[j] for j in indices) / len(indices)
        avg_acc = sum(1.0 for j in indices if correctness[j]) / len(indices)

        bins.append({
            'bin_lo': lo, 'bin_hi': hi,
            'count': len(indices),
            'avg_confidence': avg_conf,
            'avg_accuracy': avg_acc,
            'gap': abs(avg_conf - avg_acc),
        })

    # Weighted average of gaps
    ece = sum(b['count'] * b['gap'] for b in bins) / n
    return ece, bins


def confidence_gap(
    confidences: List[float],
    correctness: List[bool],
) -> dict:
    """
    Compute the confidence gap between correct and wrong answers.
    A model with genuine self-knowledge should have a large gap.

    Opus 4.6: gap = 3.7pp (decent self-knowledge)
    Qwen3-235B: gap = 0.07pp (virtually zero self-knowledge)

    Returns:
        Dict with mean_correct, mean_wrong, gap, std_all
    """
    correct_confs = [c for c, ok in zip(confidences, correctness) if ok]
    wrong_confs = [c for c, ok in zip(confidences, correctness) if not ok]
    all_confs = confidences

    mean_all = sum(all_confs) / len(all_confs) if all_confs else 0
    mean_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0
    mean_wrong = sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0

    variance = sum((c - mean_all) ** 2 for c in all_confs) / len(all_confs) if all_confs else 0
    std = math.sqrt(variance)

    return {
        'mean_all': round(mean_all, 4),
        'mean_correct': round(mean_correct, 4),
        'mean_wrong': round(mean_wrong, 4),
        'gap': round(mean_correct - mean_wrong, 4),
        'std': round(std, 4),
        'n_correct': len(correct_confs),
        'n_wrong': len(wrong_confs),
    }
