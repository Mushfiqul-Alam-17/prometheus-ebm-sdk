"""
PROMETHEUS-EBM SDK v1.1.0
=========================
Benchmarking Epistemic Metacognition in AI Models.

The world's first living benchmark for measuring what frontier AI models
do not know they do not know — with built-in anti-contamination architecture.

Evaluate whether frontier AI models can recognize the limits of their own knowledge:
not just answer questions, but understand when a question is unanswerable, ambiguous,
or self-contradictory.

Quick Start — Evaluate Models:
    from prometheus_ebm import PrometheusRunner, RunConfig

    config = RunConfig(
        mode="extended",
        models=["anthropic/claude-opus-4-6@default", "google/gemini-3.1-pro-preview"],
        provider="kaggle",
    )
    runner = PrometheusRunner(config)
    results = runner.run()

Quick Start — Generate Anti-Contamination Epochs:
    from prometheus_ebm.generator import EpochGenerator

    gen = EpochGenerator(epoch_id="v3", seed=42)
    problems = gen.generate(n_problems=1000)
    gen.save(problems, output_dir="./epoch_v3")

Quick Start — Score Your Own Data:
    from prometheus_ebm import ECIScorer, BrierDecomposition, Type2DPrime
    scorer = ECIScorer()
    eci = scorer.compute_eci(sda=0.85, ca=0.80, rp=0.75, ece=0.15, hss=0.10)

Full documentation: https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk
PyPI: https://pypi.org/project/prometheus-ebm/
"""

__version__ = "1.1.0"

from prometheus_ebm.config import (
    RunConfig,
    KAGGLE_MODEL_CATALOG,
    resolve_models_from_indices,
)
from prometheus_ebm.scorer import (
    ECIScorer,
    BrierDecomposition,
    Type2DPrime,
    ScoringResult,
    ReadinessScorer,
)
from prometheus_ebm.taxonomy import SolvabilityClass, BENCHMARK_DOMAINS
from prometheus_ebm.runner import PrometheusRunner, BenchmarkResults
from prometheus_ebm.workflow_v5 import build_v5_config, run_v5_workflow
from prometheus_ebm.providers.openai import OpenAIProvider
from prometheus_ebm.providers.openrouter import OpenRouterProvider
from prometheus_ebm.providers.anthropic import AnthropicProvider
from prometheus_ebm.providers.kaggle import KaggleProvider

__all__ = [
    # Core
    "RunConfig",
    "KAGGLE_MODEL_CATALOG",
    "resolve_models_from_indices",
    "PrometheusRunner",
    "BenchmarkResults",
    "build_v5_config",
    "run_v5_workflow",
    # Scoring
    "ECIScorer",
    "BrierDecomposition",
    "Type2DPrime",
    "ScoringResult",
    "ReadinessScorer",
    # Taxonomy
    "SolvabilityClass",
    "BENCHMARK_DOMAINS",
    # Providers
    "OpenAIProvider",
    "OpenRouterProvider",
    "AnthropicProvider",
    "KaggleProvider",
]
