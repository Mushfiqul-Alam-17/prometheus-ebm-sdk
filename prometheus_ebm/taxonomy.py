"""
4-Class Solvability Taxonomy for PROMETHEUS-EBM.

Unlike binary benchmarks (right/wrong), every problem is classified into one of 
four epistemic categories. Models are scored on whether they correctly identify 
which category a problem falls into.
"""
from enum import Enum
from typing import Optional


class SolvabilityClass(Enum):
    """The four epistemic categories in the PROMETHEUS taxonomy."""
    
    DETERMINATE = "DETERMINATE"
    """One clear, unambiguous answer exists based on the provided information."""
    
    UNDERDETERMINED = "UNDERDETERMINED"
    """Multiple valid interpretations exist; the question is genuinely ambiguous."""
    
    INSUFFICIENT = "INSUFFICIENT"
    """Critical information is missing; no definitive answer can be warranted."""
    
    CONTRADICTORY = "CONTRADICTORY"
    """The premises conflict; the model must detect the logical contradiction."""
    
    @classmethod
    def from_string(cls, s: str) -> Optional["SolvabilityClass"]:
        """Parse a solvability class from a model's response string."""
        if not s:
            return None
        normalized = s.strip().upper().replace("-", "").replace("_", "")
        mapping = {
            "DETERMINATE": cls.DETERMINATE,
            "UNDERDETERMINED": cls.UNDERDETERMINED,
            "INSUFFICIENT": cls.INSUFFICIENT,
            "CONTRADICTORY": cls.CONTRADICTORY,
            # Common model variations
            "SOLVABLE": cls.DETERMINATE,
            "UNSOLVABLE": cls.INSUFFICIENT,
            "AMBIGUOUS": cls.UNDERDETERMINED,
            "CONTRADICTION": cls.CONTRADICTORY,
            "CONTRADICTING": cls.CONTRADICTORY,
        }
        return mapping.get(normalized)
    
    @property
    def is_solvable(self) -> bool:
        """Whether this class represents a solvable problem."""
        return self == SolvabilityClass.DETERMINATE
    
    @property
    def requires_refusal(self) -> bool:
        """Whether a well-calibrated model should refuse to give a definitive answer."""
        return self in (
            SolvabilityClass.UNDERDETERMINED,
            SolvabilityClass.INSUFFICIENT,
            SolvabilityClass.CONTRADICTORY,
        )
    
    @property
    def description(self) -> str:
        """Human-readable description of this class."""
        descriptions = {
            SolvabilityClass.DETERMINATE: "One clear answer exists",
            SolvabilityClass.UNDERDETERMINED: "Multiple valid interpretations (ambiguous)",
            SolvabilityClass.INSUFFICIENT: "Critical information is missing",
            SolvabilityClass.CONTRADICTORY: "Premises conflict (contradiction)",
        }
        return descriptions[self]


# ── Domains ────────────────────────────────────────────────────────────────────

BENCHMARK_DOMAINS = [
    "medical",
    "financial", 
    "legal",
    "environmental",
    "social",
]
