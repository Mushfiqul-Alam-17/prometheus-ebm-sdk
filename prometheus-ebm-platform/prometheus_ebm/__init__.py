"""
PROMETHEUS-EBM: Epistemological Benchmark for Metacognition
A framework for evaluating how AI models handle epistemic uncertainty.

Usage:
    from prometheus_ebm import Benchmark, ECIScorer
    bench = Benchmark.load_default()
    results = bench.evaluate(model="anthropic/claude-sonnet-4-5")
    report = ECIScorer(results).summary()
"""

__version__ = "0.1.0"
__author__ = "PROMETHEUS-EBM Team"

from prometheus_ebm.taxonomy import SolvabilityClass, Problem, ModelResponse
from prometheus_ebm.dataset import Dataset
from prometheus_ebm.scorer import ECIScorer, HGIScorer
from prometheus_ebm.evaluator import Evaluator
from prometheus_ebm.stress import StressAugmenter
from prometheus_ebm.profiler import ModelProfiler
