"""
PROMETHEUS-EBM SDK
==================
Benchmarking Epistemic Metacognition in AI Models.

Evaluate whether frontier AI models can recognize the limits of their own knowledge —
not just answer questions, but understand when a question is unanswerable, ambiguous,
or self-contradictory.

Quick Start:
    from prometheus_ebm import PrometheusRunner, RunConfig

    # Compare multiple models (no API key needed on Kaggle)
    config = RunConfig(
        mode="compare",
        models=["anthropic/claude-opus-4-6@default", "google/gemini-3.1-pro-preview"],
        provider="kaggle",
    )
    runner = PrometheusRunner(config)
    results = runner.run()

    # Score your own data
    from prometheus_ebm import ECIScorer, BrierDecomposition, Type2DPrime
    scorer = ECIScorer()
    eci = scorer.compute_eci(sda=0.85, ca=0.80, rp=0.75, ece=0.15, hss=0.10)

Full documentation: https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk
"""

__version__ = "0.1.0"

from prometheus_ebm.config import RunConfig
from prometheus_ebm.scorer import ECIScorer, BrierDecomposition, Type2DPrime, ScoringResult
from prometheus_ebm.taxonomy import SolvabilityClass, BENCHMARK_DOMAINS
from prometheus_ebm.runner import PrometheusRunner, BenchmarkResults

__all__ = [
    "RunConfig",
    "PrometheusRunner",
    "BenchmarkResults",
    "ECIScorer",
    "BrierDecomposition",
    "Type2DPrime",
    "ScoringResult",
    "SolvabilityClass",
    "BENCHMARK_DOMAINS",
]
