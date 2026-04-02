"""
profiler.py — Model behavioural profiling for PROMETHEUS-EBM

Generates interpretable behavioural signatures for each model:
  - Risk profile (cautious vs assertive vs reckless)
  - Calibration profile (aligned vs overconfident)
  - Stability profile (robust vs fragile)
  - Domain/class blindspots
  - Metacognitive fingerprint

Based on Epoch-1 findings:
  Opus 4.6  → Assertive / Aligned / Fragile / Financial blindspot
  Sonnet 4.5 → Cautious / Aligned / Robust / Financial blindspot
  Gemini 3.1 → Reckless / Overconfident / Fragile / UNDERDETERMINED blindspot
  Qwen3-235B → Split / Zero self-knowledge / Fragile / CONTRADICTORY blindspot
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter, defaultdict

from prometheus_ebm.taxonomy import SolvabilityClass, ScoredResult
from prometheus_ebm.scorer import ECIScorer
from prometheus_ebm.stats import bootstrap_ci, confidence_gap


@dataclass
class ModelProfile:
    """Complete behavioural profile for a model."""
    name: str
    n_problems: int = 0

    # Scores
    eci: float = 0.0
    hgi: float = 0.0

    # Risk classification
    risk_type: str = ""          # cautious | assertive | reckless
    risk_explanation: str = ""

    # Calibration classification
    calibration_type: str = ""   # aligned | overconfident | underconfident
    calibration_explanation: str = ""
    confidence_gap_pp: float = 0.0  # percentage points between correct/wrong mean conf
    mean_confidence: float = 0.0
    overconfidence_rate: float = 0.0

    # Stability classification
    stability_type: str = ""     # robust | fragile | volatile
    stability_explanation: str = ""
    stress_drop_ds: float = 0.0  # accuracy drop under decision stress
    stress_drop_cs: float = 0.0  # accuracy drop under clarity stress

    # Blindspots
    domain_blindspot: str = ""
    domain_blindspot_accuracy: float = 0.0
    class_blindspot: str = ""
    class_blindspot_accuracy: float = 0.0

    # Strengths
    domain_strength: str = ""
    domain_strength_accuracy: float = 0.0
    class_strength: str = ""
    class_strength_accuracy: float = 0.0

    # Hallucination analysis
    hallucination_risk: str = ""  # low | moderate | high | critical
    hallucination_rate: float = 0.0

    def summary(self) -> str:
        """Human-readable profile summary."""
        lines = [
            f"{'='*60}",
            f"MODEL PROFILE: {self.name}",
            f"{'='*60}",
            f"  ECI Score:        {self.eci:.4f}",
            f"  HGI Score:        {self.hgi:.4f}",
            f"  Problems Tested:  {self.n_problems}",
            f"",
            f"  RISK TYPE:        {self.risk_type.upper()}",
            f"    {self.risk_explanation}",
            f"",
            f"  CALIBRATION:      {self.calibration_type.upper()}",
            f"    {self.calibration_explanation}",
            f"    Confidence gap:   {self.confidence_gap_pp:.1f}pp",
            f"    Overconfidence:   {self.overconfidence_rate:.1%}",
            f"",
            f"  STABILITY:        {self.stability_type.upper()}",
            f"    {self.stability_explanation}",
            f"    Decision stress:  {self.stress_drop_ds:+.1f}pp",
            f"    Clarity stress:   {self.stress_drop_cs:+.1f}pp",
            f"",
            f"  BLINDSPOTS:",
            f"    Domain: {self.domain_blindspot} ({self.domain_blindspot_accuracy:.1%})",
            f"    Class:  {self.class_blindspot} ({self.class_blindspot_accuracy:.1%})",
            f"",
            f"  STRENGTHS:",
            f"    Domain: {self.domain_strength} ({self.domain_strength_accuracy:.1%})",
            f"    Class:  {self.class_strength} ({self.class_strength_accuracy:.1%})",
            f"",
            f"  HALLUCINATION:    {self.hallucination_risk.upper()} ({self.hallucination_rate:.1%})",
            f"{'='*60}",
        ]
        return '\n'.join(lines)


class ModelProfiler:
    """
    Generates behavioural profiles from scored results.

    Usage:
        profiler = ModelProfiler()
        profile = profiler.profile(results, "anthropic/claude-opus-4-6")
        print(profile.summary())
    """

    def profile(self, results: List[ScoredResult], model_name: str) -> ModelProfile:
        """Generate a complete behavioural profile."""
        prof = ModelProfile(name=model_name, n_problems=len(results))

        if not results:
            return prof

        # Compute ECI
        scorer = ECIScorer(results)
        metrics = scorer.compute()
        prof.eci = metrics['eci']
        prof.hgi = metrics['hgi']

        # ── Risk Classification ──
        self._classify_risk(prof, results)

        # ── Calibration Classification ──
        self._classify_calibration(prof, results)

        # ── Stability Classification ──
        self._classify_stability(prof, results)

        # ── Blindspots and Strengths ──
        self._find_blindspots(prof, results)

        # ── Hallucination Analysis ──
        self._classify_hallucination(prof, results)

        return prof

    def _classify_risk(self, prof: ModelProfile, results: List[ScoredResult]):
        """Classify risk profile based on refusal and confidence patterns."""
        # Count refusals (predicting non-DETERMINATE)
        total = len(results)
        refusals = sum(1 for r in results
                       if r.predicted_class != SolvabilityClass.DETERMINATE)
        refusal_rate = refusals / total if total else 0

        # Fraction of problems that SHOULD be refused
        should_refuse = sum(1 for r in results
                            if not r.problem.solvability_class.is_solvable)
        should_refuse_rate = should_refuse / total if total else 0

        if refusal_rate < should_refuse_rate * 0.5:
            prof.risk_type = "reckless"
            prof.risk_explanation = (
                f"Refuses only {refusal_rate:.0%} of the time, but "
                f"{should_refuse_rate:.0%} of problems warrant refusal. "
                f"Answers questions it shouldn't."
            )
        elif refusal_rate > should_refuse_rate * 1.3:
            prof.risk_type = "cautious"
            prof.risk_explanation = (
                f"Refuses {refusal_rate:.0%} of the time — more than the "
                f"{should_refuse_rate:.0%} that warrant it. Errs on the side of caution."
            )
        else:
            prof.risk_type = "assertive"
            prof.risk_explanation = (
                f"Refusal rate ({refusal_rate:.0%}) roughly matches the rate of "
                f"genuinely unsolvable problems ({should_refuse_rate:.0%}). "
                f"Confident but generally appropriate."
            )

    def _classify_calibration(self, prof: ModelProfile, results: List[ScoredResult]):
        """Classify calibration profile based on confidence-accuracy alignment."""
        confs = [r.confidence for r in results]
        corrects = [r.is_correct for r in results]

        gap_data = confidence_gap(confs, corrects)
        prof.confidence_gap_pp = gap_data['gap'] * 100
        prof.mean_confidence = gap_data['mean_all']

        # Overconfidence: wrong but conf > 0.8
        overconf = sum(1 for r in results
                       if not r.is_correct and r.confidence > 0.8)
        prof.overconfidence_rate = overconf / len(results) if results else 0

        if prof.confidence_gap_pp < 1.0:
            prof.calibration_type = "zero-self-knowledge"
            prof.calibration_explanation = (
                f"Confidence gap of only {prof.confidence_gap_pp:.1f}pp — "
                f"virtually no difference between confidence when right vs wrong. "
                f"The model cannot distinguish correct from incorrect responses."
            )
        elif prof.overconfidence_rate > 0.25:
            prof.calibration_type = "overconfident"
            prof.calibration_explanation = (
                f"Overconfident-wrong on {prof.overconfidence_rate:.0%} of answers. "
                f"Mean confidence is {prof.mean_confidence:.0%} even when wrong."
            )
        elif prof.confidence_gap_pp > 5.0:
            prof.calibration_type = "well-aligned"
            prof.calibration_explanation = (
                f"Strong {prof.confidence_gap_pp:.1f}pp confidence gap between "
                f"correct and wrong answers. Good self-knowledge."
            )
        else:
            prof.calibration_type = "aligned"
            prof.calibration_explanation = (
                f"Moderate {prof.confidence_gap_pp:.1f}pp confidence gap. "
                f"Some self-knowledge but room for improvement."
            )

    def _classify_stability(self, prof: ModelProfile, results: List[ScoredResult]):
        """Classify stability based on stress-variant paired analysis."""
        # Calculate base vs stressed accuracy
        base_results = [r for r in results if r.problem.rigor_mode.value == 'base']
        ds_results = [r for r in results if r.problem.rigor_mode.value == 'decision_stress']
        cs_results = [r for r in results if r.problem.rigor_mode.value == 'clarity_stress']

        base_acc = sum(1 for r in base_results if r.is_correct) / len(base_results) if base_results else 0
        ds_acc = sum(1 for r in ds_results if r.is_correct) / len(ds_results) if ds_results else base_acc
        cs_acc = sum(1 for r in cs_results if r.is_correct) / len(cs_results) if cs_results else base_acc

        prof.stress_drop_ds = (ds_acc - base_acc) * 100
        prof.stress_drop_cs = (cs_acc - base_acc) * 100

        max_drop = min(prof.stress_drop_ds, prof.stress_drop_cs)

        if max_drop < -7:
            prof.stability_type = "fragile"
            prof.stability_explanation = (
                f"Accuracy drops up to {abs(max_drop):.0f}pp under stress. "
                f"Metacognitive performance is highly context-dependent."
            )
        elif max_drop < -3:
            prof.stability_type = "moderate"
            prof.stability_explanation = (
                f"Moderate accuracy change under stress ({max_drop:+.0f}pp). "
                f"Some sensitivity to prompting pressure."
            )
        else:
            prof.stability_type = "robust"
            prof.stability_explanation = (
                f"Minimal accuracy change under stress ({max_drop:+.0f}pp). "
                f"Metacognitive performance is stable under pressure."
            )

    def _find_blindspots(self, prof: ModelProfile, results: List[ScoredResult]):
        """Find worst and best performing domain and class."""
        # By class
        class_acc = {}
        for cls in SolvabilityClass:
            cls_results = [r for r in results if r.true_class == cls]
            if cls_results:
                acc = sum(1 for r in cls_results if r.is_correct) / len(cls_results)
                class_acc[cls.name] = acc

        if class_acc:
            worst_cls = min(class_acc, key=class_acc.get)
            best_cls = max(class_acc, key=class_acc.get)
            prof.class_blindspot = worst_cls
            prof.class_blindspot_accuracy = class_acc[worst_cls]
            prof.class_strength = best_cls
            prof.class_strength_accuracy = class_acc[best_cls]

        # By domain
        domain_acc = {}
        domains = set(r.problem.domain.value for r in results)
        for dom in domains:
            dom_results = [r for r in results if r.problem.domain.value == dom]
            if dom_results:
                acc = sum(1 for r in dom_results if r.is_correct) / len(dom_results)
                domain_acc[dom] = acc

        if domain_acc:
            worst_dom = min(domain_acc, key=domain_acc.get)
            best_dom = max(domain_acc, key=domain_acc.get)
            prof.domain_blindspot = worst_dom
            prof.domain_blindspot_accuracy = domain_acc[worst_dom]
            prof.domain_strength = best_dom
            prof.domain_strength_accuracy = domain_acc[best_dom]

    def _classify_hallucination(self, prof: ModelProfile, results: List[ScoredResult]):
        """Classify hallucination risk level."""
        impossible = [r for r in results if r.problem.solvability_class.is_impossible]
        if not impossible:
            prof.hallucination_risk = "unknown"
            prof.hallucination_rate = 0
            return

        hallucinations = [r for r in impossible if r.is_hallucination]
        rate = len(hallucinations) / len(impossible)
        prof.hallucination_rate = rate

        if rate < 0.05:
            prof.hallucination_risk = "low"
        elif rate < 0.15:
            prof.hallucination_risk = "moderate"
        elif rate < 0.25:
            prof.hallucination_risk = "high"
        else:
            prof.hallucination_risk = "critical"
