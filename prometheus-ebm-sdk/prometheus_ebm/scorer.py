"""
Scoring engine for PROMETHEUS-EBM.

Implements:
- ECI (Epistemological Calibration Index) — composite metacognition score
- HGI (Hysteresis Gap Index) — internal consistency measure
- Brier Score Decomposition — calibration breakdown (reliability/resolution/uncertainty)
- Type-2 D-Prime — signal detection for metacognitive discrimination
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class ScoringResult:
    """Complete scoring result for a single model."""
    model: str
    n_items: int
    eci: float
    sda: float
    ca: float
    rp: float
    ece: float
    hss: float
    hgi: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    d_prime: float
    hit_rate: float
    false_alarm_rate: float
    overconfidence_gap: float


class ECIScorer:
    """
    Epistemological Calibration Index scorer.
    
    ECI = w_sda*SDA + w_ca*CA + w_rp*RP + w_ece*(1-ECE) + w_hss*(1-HSS)
    
    Component weights:
        SDA (30%): Solvability Detection Accuracy
        CA  (25%): Conditional Accuracy on determinate tasks
        RP  (20%): Refusal Precision on impossible tasks
        ECE (15%): Expected Calibration Error (inverted: lower is better)
        HSS (10%): Hallucination Sensitivity Score (inverted: lower is better)
    """
    
    WEIGHTS = {
        'sda': 0.30,
        'ca': 0.25,
        'rp': 0.20,
        'ece': 0.15,
        'hss': 0.10,
    }
    
    @staticmethod
    def compute_hgi(confidences: List[float], correctness: List[bool], 
                    solvability_correct: List[bool]) -> float:
        """
        Hysteresis Gap Index: measures internal inconsistency.
        HGI = mean(abs(conf - (is_correct + solv_ok)/2.0))
        """
        if not confidences:
            return 0.0
        
        gaps = [
            abs(float(c) - (float(corr) + float(solv)) / 2.0)
            for c, corr, solv in zip(confidences, correctness, solvability_correct)
        ]
        return float(np.mean(gaps))
    
    @staticmethod
    def compute_sda(predicted_classes: List[str], true_classes: List[str]) -> float:
        """Solvability Detection Accuracy: fraction of correct class predictions."""
        if not predicted_classes:
            return 0.0
        correct = sum(1 for p, t in zip(predicted_classes, true_classes) 
                      if p and t and p.upper() == t.upper())
        return correct / len(predicted_classes)
    
    @staticmethod
    def compute_ca(answers_correct: List[bool], true_classes: List[str]) -> float:
        """Conditional Accuracy: accuracy on DETERMINATE problems only."""
        det_correct = [c for c, cls in zip(answers_correct, true_classes)
                       if cls and cls.upper() == "DETERMINATE"]
        if not det_correct:
            return 0.0
        return sum(det_correct) / len(det_correct)
    
    @staticmethod
    def compute_rp(predicted_classes: List[str], true_classes: List[str]) -> float:
        """Refusal Precision: when model says 'not determinate', is it correct?"""
        refusals = [(p, t) for p, t in zip(predicted_classes, true_classes)
                    if p and p.upper() != "DETERMINATE"]
        if not refusals:
            return 0.0
        correct_refusals = sum(1 for p, t in refusals if t and t.upper() != "DETERMINATE")
        return correct_refusals / len(refusals)
    
    @staticmethod
    def compute_ece(confidences: List[float], correctness: List[bool], n_bins: int = 10) -> float:
        """Expected Calibration Error across confidence bins."""
        if not confidences:
            return 0.0
        confs = np.array(confidences)
        correct = np.array(correctness, dtype=float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for k in range(n_bins):
            mask = (confs >= bin_edges[k]) & (confs < bin_edges[k + 1])
            if k == n_bins - 1:
                mask = mask | (confs == bin_edges[k + 1])
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue
            avg_conf = float(np.mean(confs[mask]))
            avg_acc = float(np.mean(correct[mask]))
            ece += (n_k / len(confs)) * abs(avg_acc - avg_conf)
        return ece
    
    @staticmethod
    def compute_hss(
        answers_correct: List[bool],
        true_classes: List[str],
        confidences: Optional[List[float]] = None,
        confidence_threshold: float = 0.6,
        include_underdetermined: bool = False,
    ) -> float:
        """
        Hallucination Sensitivity Score for impossible tasks.

        Notebook V5 parity path (when confidences are provided):
            hss = mean(incorrect AND confidence > threshold) on impossible classes.

        Backward-compatible path (no confidences):
            hss = 1 - mean(is_correct) on impossible classes.
        """
        impossible_labels = {"INSUFFICIENT", "CONTRADICTORY"}
        if include_underdetermined:
            impossible_labels.add("UNDERDETERMINED")

        impossible_idx = [
            i for i, t in enumerate(true_classes)
            if t and str(t).upper() in impossible_labels and i < len(answers_correct)
        ]
        if not impossible_idx:
            return 0.0

        if confidences is None:
            correct_on_impossible = sum(1 for i in impossible_idx if bool(answers_correct[i]))
            return 1.0 - (correct_on_impossible / len(impossible_idx))

        high_conf_incorrect = 0
        for i in impossible_idx:
            if i >= len(confidences):
                continue
            if (not bool(answers_correct[i])) and float(confidences[i]) > confidence_threshold:
                high_conf_incorrect += 1

        return float(high_conf_incorrect) / float(len(impossible_idx))
    
    def compute_eci(self, sda: float, ca: float, rp: float, 
                    ece: float, hss: float) -> float:
        """Compute the composite ECI score."""
        return (
            self.WEIGHTS['sda'] * sda +
            self.WEIGHTS['ca'] * ca +
            self.WEIGHTS['rp'] * rp +
            self.WEIGHTS['ece'] * (1.0 - ece) +
            self.WEIGHTS['hss'] * (1.0 - hss)
        )


class BrierDecomposition:
    """
    Brier Score Decomposition into Reliability, Resolution, and Uncertainty.
    
    - Reliability: How well predicted probabilities match observed frequencies (lower = better)
    - Resolution: How much confidence varies with correctness (higher = better discrimination)
    - Uncertainty: Baseline difficulty of the dataset (constant for a given dataset)
    
    Brier = Reliability - Resolution + Uncertainty
    """
    
    @staticmethod
    def compute(confidences: List[float], correctness: List[bool], 
                n_bins: int = 10) -> Dict[str, float]:
        """Compute Brier score decomposition."""
        confs = np.array(confidences, dtype=float)
        correct = np.array(correctness, dtype=float)
        
        if len(confs) == 0:
            return {'brier': float('nan'), 'reliability': float('nan'),
                    'resolution': float('nan'), 'uncertainty': float('nan')}
        
        brier = float(np.mean((confs - correct) ** 2))
        base_rate = float(np.mean(correct))
        uncertainty = base_rate * (1 - base_rate)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        reliability = 0.0
        resolution = 0.0
        
        for k in range(n_bins):
            mask = (confs >= bin_edges[k]) & (confs < bin_edges[k + 1])
            if k == n_bins - 1:
                mask = mask | (confs == bin_edges[k + 1])
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue
            avg_conf_k = float(np.mean(confs[mask]))
            avg_correct_k = float(np.mean(correct[mask]))
            reliability += n_k * (avg_correct_k - avg_conf_k) ** 2
            resolution += n_k * (avg_correct_k - base_rate) ** 2
        
        n_total = len(confs)
        reliability /= n_total
        resolution /= n_total
        
        return {
            'brier': round(brier, 6),
            'reliability': round(reliability, 6),
            'resolution': round(resolution, 6),
            'uncertainty': round(uncertainty, 6),
        }


class Type2DPrime:
    """
    Type-2 Signal Detection (D-Prime) for metacognitive discrimination.
    
    Measures how well a model's internal confidence signal distinguishes
    between its own correct and incorrect answers.
    
    - Hit: confident AND correct
    - False Alarm: confident AND incorrect  
    - D' = Z(hit_rate) - Z(false_alarm_rate)
    
    D' ≈ 0: confidence is uninformative (random noise)
    D' > 1: strong metacognitive discrimination
    D' > 2: excellent metacognitive awareness
    """
    
    @staticmethod
    def compute(confidences: List[float], correctness: List[bool],
                threshold: float = 0.7) -> Dict[str, float]:
        """Compute Type-2 D-Prime."""
        from scipy.stats import norm
        
        confs = np.array(confidences, dtype=float)
        correct = np.array(correctness, dtype=bool)
        confident = confs >= threshold
        
        n_correct = int(np.sum(correct))
        n_incorrect = int(np.sum(~correct))
        
        if n_correct == 0 or n_incorrect == 0:
            return {'d_prime': float('nan'), 'hit_rate': float('nan'),
                    'false_alarm_rate': float('nan'), 'threshold': threshold}
        
        hit_rate = float(np.sum(confident & correct)) / n_correct
        false_alarm_rate = float(np.sum(confident & ~correct)) / n_incorrect
        
        # Log-linear correction for extreme rates
        hit_adj = (hit_rate * n_correct + 0.5) / (n_correct + 1)
        fa_adj = (false_alarm_rate * n_incorrect + 0.5) / (n_incorrect + 1)
        
        d_prime = float(norm.ppf(hit_adj) - norm.ppf(fa_adj))
        
        return {
            'd_prime': round(d_prime, 4),
            'hit_rate': round(hit_rate, 4),
            'false_alarm_rate': round(false_alarm_rate, 4),
            'threshold': threshold,
        }
