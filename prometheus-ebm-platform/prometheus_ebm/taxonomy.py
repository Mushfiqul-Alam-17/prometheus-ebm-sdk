"""
taxonomy.py — Core type definitions for PROMETHEUS-EBM

The 4-class solvability taxonomy is our primary innovation.
Every other benchmark uses binary (solvable/unsolvable).
We prove that UNDERDETERMINED, INSUFFICIENT, and CONTRADICTORY
require fundamentally different metacognitive skills.

Evidence: Gemini 3.1 scores 97.3% on INSUFFICIENT but 19.2% on
UNDERDETERMINED — a 78pp gap invisible in binary classification.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class SolvabilityClass(Enum):
    """
    The four epistemic states a problem can occupy.

    DETERMINATE:     Exactly one correct answer derivable from given info.
    UNDERDETERMINED: Multiple valid interpretations; ambiguity is inherent.
    INSUFFICIENT:    Critical information is missing; cannot be solved.
    CONTRADICTORY:   Internal contradictions make any answer invalid.

    This taxonomy is the core differentiator of PROMETHEUS-EBM.
    Binary benchmarks (UnsolvableQA, MathTrap300, AbstentionBench)
    collapse U, I, C into a single "unsolvable" class, losing the
    78pp discrimination power we demonstrate.
    """
    DETERMINATE = "D"
    UNDERDETERMINED = "U"
    INSUFFICIENT = "I"
    CONTRADICTORY = "C"

    @classmethod
    def from_string(cls, s: str) -> 'SolvabilityClass':
        """Parse solvability class from various string formats."""
        s = s.strip().upper()
        mapping = {
            'D': cls.DETERMINATE, 'DETERMINATE': cls.DETERMINATE, 'DET': cls.DETERMINATE,
            'U': cls.UNDERDETERMINED, 'UNDERDETERMINED': cls.UNDERDETERMINED, 'UND': cls.UNDERDETERMINED,
            'I': cls.INSUFFICIENT, 'INSUFFICIENT': cls.INSUFFICIENT, 'INS': cls.INSUFFICIENT,
            'C': cls.CONTRADICTORY, 'CONTRADICTORY': cls.CONTRADICTORY, 'CON': cls.CONTRADICTORY,
        }
        if s in mapping:
            return mapping[s]
        # Fuzzy matching
        sl = s.lower()
        if 'under' in sl:
            return cls.UNDERDETERMINED
        if 'insuff' in sl or 'missing' in sl:
            return cls.INSUFFICIENT
        if 'contrad' in sl or 'inconsist' in sl:
            return cls.CONTRADICTORY
        if 'determin' in sl:
            return cls.DETERMINATE
        raise ValueError(f"Cannot parse solvability class from: {s!r}")

    @property
    def is_solvable(self) -> bool:
        return self == SolvabilityClass.DETERMINATE

    @property
    def is_impossible(self) -> bool:
        return self in (SolvabilityClass.INSUFFICIENT, SolvabilityClass.CONTRADICTORY)

    @property
    def is_ambiguous(self) -> bool:
        return self == SolvabilityClass.UNDERDETERMINED


class Domain(Enum):
    """Professional domains covered by PROMETHEUS-EBM."""
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"

    @classmethod
    def from_string(cls, s: str) -> 'Domain':
        s = s.strip().lower()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown domain: {s!r}")


class RigorMode(Enum):
    """Prompting rigor modes for stress testing."""
    BASE = "base"
    DECISION_STRESS = "decision_stress"
    CLARITY_STRESS = "clarity_stress"


@dataclass
class Problem:
    """
    A single evaluation problem in the PROMETHEUS-EBM benchmark.

    Each problem has a known solvability class, domain, and
    ground truth answer (or explicit "no valid answer" for I/C).
    """
    problem_id: str
    domain: Domain
    solvability_class: SolvabilityClass
    system_prompt: str
    user_prompt: str
    ground_truth_answer: Optional[str] = None
    branching_factor: int = 2
    rigor_mode: RigorMode = RigorMode.BASE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def problem_class(self) -> str:
        """Backward-compatible string representation."""
        return self.solvability_class.name

    def to_dict(self) -> dict:
        return {
            'problem_id': self.problem_id,
            'domain': self.domain.value,
            'problem_class': self.solvability_class.name,
            'correct_solvability_class': self.solvability_class.name,
            'system': self.system_prompt,
            'user': self.user_prompt,
            'ground_truth_answer': self.ground_truth_answer,
            'branching_factor': self.branching_factor,
            'rigor_mode': self.rigor_mode.value,
        }


@dataclass
class ModelResponse:
    """
    Parsed response from a model for a single problem.
    """
    problem_id: str
    model_name: str
    raw_response: str
    final_answer: Optional[str] = None
    solvability_estimate: Optional[SolvabilityClass] = None
    confidence: float = 0.5
    justification_type: Optional[str] = None
    reasoning: Optional[str] = None
    parse_success: bool = False
    latency_ms: float = 0.0

    # Scoring fields (populated by scorer)
    is_correct: Optional[bool] = None
    correctness_method: Optional[str] = None
    failure_mode: Optional[str] = None
    failure_severity: Optional[str] = None


@dataclass
class ScoredResult:
    """
    Full scored result combining problem, response, and metrics.
    """
    problem: Problem
    response: ModelResponse

    @property
    def is_correct(self) -> bool:
        return self.response.is_correct or False

    @property
    def confidence(self) -> float:
        return self.response.confidence

    @property
    def true_class(self) -> SolvabilityClass:
        return self.problem.solvability_class

    @property
    def predicted_class(self) -> Optional[SolvabilityClass]:
        return self.response.solvability_estimate

    @property
    def solvability_correct(self) -> bool:
        return self.true_class == self.predicted_class

    @property
    def is_hallucination(self) -> bool:
        """Confident wrong answer on an impossible problem."""
        return (
            self.problem.solvability_class.is_impossible
            and not self.is_correct
            and self.confidence > 0.6
        )


# ── Failure Mode Taxonomy ───────────────────────────────────────────────────

class FailureMode(Enum):
    """
    10-category failure mode taxonomy.
    Categorizes HOW a model fails, not just that it failed.
    """
    # On DETERMINATE problems
    WRONG_ANSWER = "wrong_answer"                    # Knew solvable, got wrong answer
    FALSE_UNSOLVABLE = "false_unsolvable"             # Thought solvable problem was unsolvable

    # On UNDERDETERMINED problems
    FALSE_DETERMINISM = "false_determinism"           # Treated ambiguous as having one answer
    WRONG_UNSOLVABLE_TYPE = "wrong_unsolvable_type"   # Right it's unsolvable, wrong type
    MISSED_AMBIGUITY = "missed_ambiguity"             # Failed to detect ambiguity

    # On INSUFFICIENT problems
    HALLUCINATION_ON_MISSING = "hallucination_missing"  # Made up answer with missing data
    MISSED_INSUFFICIENCY = "missed_insufficiency"        # Failed to detect missing info

    # On CONTRADICTORY problems
    HALLUCINATION_ON_CONTRADICTION = "hallucination_contradiction"  # Ignored contradiction
    MISSED_CONTRADICTION = "missed_contradiction"                    # Failed to detect contradiction

    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Severity based on confidence when failing."""
    HIGH_CONFIDENCE = "high"       # conf > 0.8 — dangerous
    MODERATE_CONFIDENCE = "moderate"  # 0.5 < conf <= 0.8
    LOW_CONFIDENCE = "low"          # conf <= 0.5 — at least uncertain


def classify_failure(
    problem: Problem,
    response: ModelResponse,
) -> tuple:
    """
    Classify a failure into (FailureMode, FailureSeverity).
    Returns (None, None) if the response is correct.
    """
    if response.is_correct:
        return None, None

    cls = problem.solvability_class
    pred = response.solvability_estimate

    # Determine severity
    if response.confidence > 0.8:
        severity = FailureSeverity.HIGH_CONFIDENCE
    elif response.confidence > 0.5:
        severity = FailureSeverity.MODERATE_CONFIDENCE
    else:
        severity = FailureSeverity.LOW_CONFIDENCE

    # Determine mode
    if cls == SolvabilityClass.DETERMINATE:
        if pred == SolvabilityClass.DETERMINATE:
            mode = FailureMode.WRONG_ANSWER
        else:
            mode = FailureMode.FALSE_UNSOLVABLE

    elif cls == SolvabilityClass.UNDERDETERMINED:
        if pred == SolvabilityClass.DETERMINATE:
            mode = FailureMode.FALSE_DETERMINISM
        elif pred in (SolvabilityClass.INSUFFICIENT, SolvabilityClass.CONTRADICTORY):
            mode = FailureMode.WRONG_UNSOLVABLE_TYPE
        else:
            mode = FailureMode.MISSED_AMBIGUITY

    elif cls == SolvabilityClass.INSUFFICIENT:
        if pred == SolvabilityClass.DETERMINATE:
            mode = FailureMode.HALLUCINATION_ON_MISSING
        elif pred in (SolvabilityClass.UNDERDETERMINED, SolvabilityClass.CONTRADICTORY):
            mode = FailureMode.WRONG_UNSOLVABLE_TYPE
        else:
            mode = FailureMode.MISSED_INSUFFICIENCY

    elif cls == SolvabilityClass.CONTRADICTORY:
        if pred == SolvabilityClass.DETERMINATE:
            mode = FailureMode.HALLUCINATION_ON_CONTRADICTION
        elif pred in (SolvabilityClass.UNDERDETERMINED, SolvabilityClass.INSUFFICIENT):
            mode = FailureMode.WRONG_UNSOLVABLE_TYPE
        else:
            mode = FailureMode.MISSED_CONTRADICTION
    else:
        mode = FailureMode.UNKNOWN

    return mode, severity
