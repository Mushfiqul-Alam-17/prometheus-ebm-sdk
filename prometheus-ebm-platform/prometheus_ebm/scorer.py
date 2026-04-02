"""
scorer.py — Core scoring engine for PROMETHEUS-EBM

Computes the 7-metric evaluation suite:
  ECI  = Epistemological Calibration Index (composite)
  SDA  = Solvability Detection Accuracy
  CA   = Conditional Accuracy
  RP   = Refusal Precision
  ECE  = Expected Calibration Error
  HSS  = Hallucination Severity Score
  HGI  = Hysteresis Gap Index

No other benchmark computes all 7 simultaneously.
KalshiBench uses ECE alone. DMC uses a single scalar.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from prometheus_ebm.taxonomy import (
    SolvabilityClass, Problem, ModelResponse, ScoredResult,
    FailureMode, FailureSeverity, classify_failure,
)
from prometheus_ebm.stats import (
    bootstrap_ci, expected_calibration_error, confidence_gap, cohens_h,
)


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_response(raw: str) -> dict:
    """
    Parse a model's structured response into components.
    Expects format with FINAL_ANSWER, SOLVABILITY_CLASS, CONFIDENCE, etc.
    """
    text = str(raw or '').strip()

    def extract(field: str) -> Optional[str]:
        pat = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    final_answer = extract('FINAL_ANSWER')
    solv_raw = extract('SOLVABILITY_CLASS')
    conf_raw = extract('CONFIDENCE')
    just = extract('JUSTIFICATION_TYPE')
    reasoning = extract('REASONING')

    # Parse solvability
    solvability = None
    if solv_raw:
        try:
            solvability = SolvabilityClass.from_string(solv_raw)
        except ValueError:
            pass

    # Parse confidence
    confidence = 0.5
    if conf_raw:
        nums = re.findall(r'\d+\.?\d*', conf_raw)
        if nums:
            v = float(nums[0])
            confidence = v / 100.0 if v > 1.0 else v
            confidence = max(0.0, min(1.0, confidence))

    return {
        'final_answer': final_answer,
        'solvability_estimate': solvability,
        'confidence': confidence,
        'justification_type': just,
        'reasoning': reasoning,
        'parse_success': final_answer is not None and solvability is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CORRECTNESS EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_correctness(
    problem: Problem,
    response: ModelResponse,
) -> Tuple[bool, str]:
    """
    Evaluate whether a response is correct given the problem's ground truth.

    For DETERMINATE: keyword matching against ground truth.
    For UNDERDETERMINED: must acknowledge ambiguity.
    For INSUFFICIENT: must refuse or identify missing info.
    For CONTRADICTORY: must identify contradiction.

    Returns:
        (is_correct, method_used)
    """
    cls = problem.solvability_class
    pred = response.solvability_estimate
    answer = response.final_answer

    if cls == SolvabilityClass.DETERMINATE:
        if answer is None or problem.ground_truth_answer in (None, '', 'None'):
            return False, 'missing_answer_or_gt'
        gt = str(problem.ground_truth_answer).lower()
        ans = str(answer).lower()
        key_terms = [t for t in gt.split() if len(t) > 4]
        if not key_terms:
            return gt in ans, 'substring_match'
        hits = sum(1 for t in key_terms if t in ans)
        return hits >= max(1, len(key_terms) // 3), 'keyword_fallback'

    elif cls == SolvabilityClass.UNDERDETERMINED:
        if pred == SolvabilityClass.UNDERDETERMINED:
            return True, 'correct_class_identified'
        if answer and any(w in str(answer).lower()
                          for w in ['multiple', 'depends', 'ambiguous', 'several', 'various']):
            return True, 'ambiguity_acknowledged'
        return False, 'failed_underdetermined'

    elif cls == SolvabilityClass.INSUFFICIENT:
        if pred == SolvabilityClass.INSUFFICIENT:
            return True, 'correct_class_identified'
        if answer and any(w in str(answer).lower()
                          for w in ['cannot', 'insufficient', 'not enough', 'missing', 'unable']):
            return True, 'correct_refusal'
        return False, 'hallucinated_on_insufficient'

    elif cls == SolvabilityClass.CONTRADICTORY:
        if pred == SolvabilityClass.CONTRADICTORY:
            return True, 'correct_class_identified'
        if answer and any(w in str(answer).lower()
                          for w in ['contradict', 'inconsistent', 'impossible', 'conflict']):
            return True, 'contradiction_identified'
        return False, 'failed_contradiction'

    return False, 'unknown_class'


# ═══════════════════════════════════════════════════════════════════════════
# ECI SCORER — The composite metric
# ═══════════════════════════════════════════════════════════════════════════

class ECIScorer:
    """
    Epistemological Calibration Index (ECI) Scorer.

    ECI = 0.30·SDA + 0.25·CA + 0.20·RP + 0.15·(1-ECE) + 0.10·(1-HSS)

    Components:
        SDA: Solvability Detection Accuracy — % of correct class predictions
        CA:  Conditional Accuracy — % correct on DETERMINATE problems
        RP:  Refusal Precision — when model refuses, was refusal appropriate?
        ECE: Expected Calibration Error — confidence-accuracy alignment
        HSS: Hallucination Severity Score — confident errors on impossible problems
    """

    # Weights for the composite score
    W_SDA = 0.30
    W_CA = 0.25
    W_RP = 0.20
    W_ECE = 0.15
    W_HSS = 0.10

    def __init__(self, results: List[ScoredResult]):
        self.results = results
        self._metrics = None

    def compute(self) -> Dict[str, float]:
        """Compute all 7 metrics."""
        if self._metrics is not None:
            return self._metrics

        n = len(self.results)
        if n == 0:
            return {'eci': 0, 'sda': 0, 'ca': 0, 'rp': 0, 'ece': 0, 'hss': 0, 'hgi': 0, 'n': 0}

        # SDA: Solvability Detection Accuracy
        sda_correct = sum(1 for r in self.results if r.solvability_correct)
        sda = sda_correct / n

        # CA: Conditional Accuracy (on DETERMINATE problems only)
        det_results = [r for r in self.results
                       if r.problem.solvability_class == SolvabilityClass.DETERMINATE]
        ca = sum(1 for r in det_results if r.is_correct) / len(det_results) if det_results else 0

        # RP: Refusal Precision
        # When model says "not DETERMINATE", is it correct to do so?
        refusals = [r for r in self.results
                    if r.predicted_class != SolvabilityClass.DETERMINATE]
        rp = sum(1 for r in refusals
                 if r.problem.solvability_class != SolvabilityClass.DETERMINATE
                 ) / len(refusals) if refusals else 0

        # ECE: Expected Calibration Error
        confs = [r.confidence for r in self.results]
        corrects = [r.is_correct for r in self.results]
        ece, ece_bins = expected_calibration_error(confs, corrects)

        # HSS: Hallucination Severity Score
        impossible = [r for r in self.results if r.problem.solvability_class.is_impossible]
        if impossible:
            hallucinations = [r for r in impossible if r.is_hallucination]
            hss = len(hallucinations) / len(impossible)
        else:
            hss = 0

        # ECI: Composite
        eci = (
            self.W_SDA * sda
            + self.W_CA * ca
            + self.W_RP * rp
            + self.W_ECE * (1 - ece)
            + self.W_HSS * (1 - hss)
        )

        # HGI: Hysteresis Gap Index
        hgi = HGIScorer(self.results).compute()

        self._metrics = {
            'n': n,
            'eci': round(eci, 6),
            'sda': round(sda, 6),
            'ca': round(ca, 6),
            'rp': round(rp, 6),
            'ece': round(ece, 6),
            'hss': round(hss, 6),
            'hgi': round(hgi, 6),
        }
        return self._metrics

    def summary(self) -> str:
        """Human-readable summary string."""
        m = self.compute()
        lines = [
            f"PROMETHEUS-EBM Scores (n={m['n']})",
            f"  ECI  = {m['eci']:.4f}  (Epistemological Calibration Index)",
            f"  SDA  = {m['sda']:.4f}  (Solvability Detection Accuracy)",
            f"  CA   = {m['ca']:.4f}  (Conditional Accuracy)",
            f"  RP   = {m['rp']:.4f}  (Refusal Precision)",
            f"  ECE  = {m['ece']:.4f}  (Expected Calibration Error, lower=better)",
            f"  HSS  = {m['hss']:.4f}  (Hallucination Severity, lower=better)",
            f"  HGI  = {m['hgi']:.4f}  (Hysteresis Gap Index, lower=better)",
        ]
        return '\n'.join(lines)

    def per_class_accuracy(self) -> Dict[str, dict]:
        """Accuracy breakdown by solvability class with bootstrap CIs."""
        out = {}
        for cls in SolvabilityClass:
            vals = [1.0 if r.is_correct else 0.0
                    for r in self.results if r.true_class == cls]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                out[cls.name] = {
                    'accuracy': round(mean, 4),
                    'ci_lower': round(lo, 4),
                    'ci_upper': round(hi, 4),
                    'n': len(vals),
                }
        return out

    def per_domain_accuracy(self) -> Dict[str, dict]:
        """Accuracy breakdown by domain with bootstrap CIs."""
        out = {}
        domains = set(r.problem.domain.value for r in self.results)
        for dom in sorted(domains):
            vals = [1.0 if r.is_correct else 0.0
                    for r in self.results if r.problem.domain.value == dom]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                out[dom] = {
                    'accuracy': round(mean, 4),
                    'ci_lower': round(lo, 4),
                    'ci_upper': round(hi, 4),
                    'n': len(vals),
                }
        return out

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Solvability confusion matrix: true class → predicted class."""
        matrix = {}
        for true_cls in SolvabilityClass:
            matrix[true_cls.name] = {}
            for pred_cls in list(SolvabilityClass) + [None]:
                pred_name = pred_cls.name if pred_cls else 'None'
                count = sum(
                    1 for r in self.results
                    if r.true_class == true_cls and r.predicted_class == pred_cls
                )
                matrix[true_cls.name][pred_name] = count
        return matrix

    def failure_taxonomy(self) -> Dict[str, int]:
        """Count failures by (FailureMode, FailureSeverity) combinations."""
        counts = defaultdict(int)
        for r in self.results:
            if r.is_correct:
                continue
            mode, severity = classify_failure(r.problem, r.response)
            if mode:
                key = f"{mode.value}|{severity.value if severity else 'unknown'}"
                counts[key] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def calibration_data(self) -> dict:
        """Confidence distribution and calibration analysis."""
        confs = [r.confidence for r in self.results]
        corrects = [r.is_correct for r in self.results]
        return confidence_gap(confs, corrects)


# ═══════════════════════════════════════════════════════════════════════════
# HGI SCORER — Hysteresis Gap Index
# ═══════════════════════════════════════════════════════════════════════════

class HGIScorer:
    """
    Hysteresis Gap Index (HGI) — three-way coherence metric.

    Measures internal consistency between:
    1. Confidence level
    2. Correctness of answer
    3. Accuracy of solvability classification

    A model with perfect metacognition would have:
    - High confidence when correct AND solvability-correct
    - Low confidence when wrong OR solvability-wrong

    HGI = 0 means perfect coherence. Higher = more internally inconsistent.

    No prior metric captures this three-way relationship.
    """

    def __init__(self, results: List[ScoredResult]):
        self.results = results

    def compute(self) -> float:
        """Compute HGI score."""
        if not self.results:
            return 0.0

        gaps = []
        for r in self.results:
            # Expected confidence: should be high when both correct and class-correct
            correct_signal = 1.0 if r.is_correct else 0.0
            class_signal = 1.0 if r.solvability_correct else 0.0
            expected_conf = (correct_signal + class_signal) / 2.0

            # Actual confidence
            actual_conf = r.confidence

            # Gap
            gaps.append(abs(actual_conf - expected_conf))

        return round(sum(gaps) / len(gaps), 6) if gaps else 0.0
