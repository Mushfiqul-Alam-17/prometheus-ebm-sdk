"""Unit tests for PROMETHEUS-EBM scoring engine."""
import pytest
import math


def test_brier_perfect_calibration():
    """A perfectly calibrated model should have Brier = 0 and Reliability = 0."""
    from prometheus_ebm.scorer import BrierDecomposition
    
    # Perfect: confidence = 1.0 when correct, 0.0 when wrong
    confs = [1.0, 1.0, 0.0, 1.0, 0.0]
    correct = [True, True, False, True, False]
    
    result = BrierDecomposition.compute(confs, correct)
    assert result['brier'] == 0.0
    assert result['reliability'] == 0.0


def test_brier_worst_calibration():
    """A maximally miscalibrated model should have high Brier score."""
    from prometheus_ebm.scorer import BrierDecomposition
    
    # Worst: confidence = 1.0 when wrong, 0.0 when correct
    confs = [0.0, 0.0, 1.0, 0.0, 1.0]
    correct = [True, True, False, True, False]
    
    result = BrierDecomposition.compute(confs, correct)
    assert result['brier'] == 1.0


def test_brier_empty_input():
    """Empty input should return NaN."""
    from prometheus_ebm.scorer import BrierDecomposition
    
    result = BrierDecomposition.compute([], [])
    assert math.isnan(result['brier'])


def test_dprime_perfect_discrimination():
    """A model with perfect metacognition should have high D-Prime."""
    from prometheus_ebm.scorer import Type2DPrime
    
    # Perfect: high confidence when correct, low when wrong
    confs = [0.95, 0.90, 0.85, 0.80, 0.10, 0.15, 0.20, 0.05]
    correct = [True, True, True, True, False, False, False, False]
    
    result = Type2DPrime.compute(confs, correct, threshold=0.7)
    assert result['d_prime'] > 2.0  # Excellent discrimination
    assert result['hit_rate'] == 1.0
    assert result['false_alarm_rate'] == 0.0


def test_dprime_random_confidence():
    """A model with random confidence should have D-Prime near 0."""
    from prometheus_ebm.scorer import Type2DPrime
    import random
    random.seed(42)
    
    # Random: confidence is unrelated to correctness
    n = 200
    confs = [random.random() for _ in range(n)]
    correct = [random.choice([True, False]) for _ in range(n)]
    
    result = Type2DPrime.compute(confs, correct, threshold=0.5)
    assert abs(result['d_prime']) < 0.5  # Near zero


def test_dprime_no_correct():
    """If no answers are correct, D-Prime should be NaN."""
    from prometheus_ebm.scorer import Type2DPrime
    
    confs = [0.9, 0.8, 0.7]
    correct = [False, False, False]
    
    result = Type2DPrime.compute(confs, correct)
    assert math.isnan(result['d_prime'])


def test_eci_weights_sum_to_one():
    """ECI component weights must sum to 1.0."""
    from prometheus_ebm.scorer import ECIScorer
    
    total = sum(ECIScorer.WEIGHTS.values())
    assert abs(total - 1.0) < 1e-10


def test_eci_max_score():
    """Perfect scores across all components should give ECI = 1.0."""
    from prometheus_ebm.scorer import ECIScorer
    
    scorer = ECIScorer()
    eci = scorer.compute_eci(sda=1.0, ca=1.0, rp=1.0, ece=0.0, hss=0.0)
    assert abs(eci - 1.0) < 1e-10


def test_eci_min_score():
    """Worst scores across all components should give ECI = 0.0."""
    from prometheus_ebm.scorer import ECIScorer
    
    scorer = ECIScorer()
    eci = scorer.compute_eci(sda=0.0, ca=0.0, rp=0.0, ece=1.0, hss=1.0)
    assert abs(eci - 0.0) < 1e-10


def test_taxonomy_from_string():
    """Taxonomy parser should handle common model output variations."""
    from prometheus_ebm.taxonomy import SolvabilityClass
    
    assert SolvabilityClass.from_string("DETERMINATE") == SolvabilityClass.DETERMINATE
    assert SolvabilityClass.from_string("Underdetermined") == SolvabilityClass.UNDERDETERMINED
    assert SolvabilityClass.from_string("INSUFFICIENT") == SolvabilityClass.INSUFFICIENT
    assert SolvabilityClass.from_string("CONTRADICTORY") == SolvabilityClass.CONTRADICTORY
    assert SolvabilityClass.from_string("") is None
    assert SolvabilityClass.from_string(None) is None


def test_taxonomy_properties():
    """Taxonomy properties should correctly classify solvability."""
    from prometheus_ebm.taxonomy import SolvabilityClass
    
    assert SolvabilityClass.DETERMINATE.is_solvable is True
    assert SolvabilityClass.UNDERDETERMINED.is_solvable is False
    assert SolvabilityClass.INSUFFICIENT.requires_refusal is True
    assert SolvabilityClass.CONTRADICTORY.requires_refusal is True


def test_config_validation():
    """RunConfig should reject invalid configurations."""
    from prometheus_ebm.config import RunConfig
    
    # No models
    config = RunConfig(mode="compare", models=[])
    with pytest.raises(ValueError, match="At least one model"):
        config.validate()
    
    # Deep probe with multiple models
    config = RunConfig(mode="deep_probe", models=["a", "b"])
    with pytest.raises(ValueError, match="deep_probe mode supports only 1 model"):
        config.validate()
    
    # Non-kaggle without API key
    config = RunConfig(mode="compare", models=["a"], provider="anthropic")
    with pytest.raises(ValueError, match="API key required"):
        config.validate()


def test_config_summary():
    """RunConfig summary should be human-readable."""
    from prometheus_ebm.config import RunConfig
    
    config = RunConfig(
        mode="compare", 
        models=["claude-opus", "gemini"],
        provider="kaggle",
    )
    summary = config.summary()
    assert "PROMETHEUS-EBM" in summary
    assert "standard" in summary
    assert "claude-opus" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
