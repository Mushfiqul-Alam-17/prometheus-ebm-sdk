# PROMETHEUS-EBM

**Epistemological Benchmark for Metacognition**

> Does the model know what it knows, what it doesn't know, and whether the question even has an answer?

PROMETHEUS-EBM is a framework for evaluating how AI models handle epistemic uncertainty. Unlike traditional benchmarks that measure correctness, PROMETHEUS measures **metacognition**: the ability to reason about one's own reasoning.

## Key Innovation: 4-Class Solvability Taxonomy

Every other metacognition benchmark uses binary classification (solvable/unsolvable). PROMETHEUS uses four classes:

| Class | What It Tests | Example |
|-------|---------------|---------|
| **Determinate** | Can the model solve a well-defined problem? | "What is 2+2?" |
| **Underdetermined** | Can the model recognize ambiguity? | "What is the best treatment?" (without specifying condition) |
| **Insufficient** | Can the model detect missing information? | "Calculate the ROI" (without providing costs) |
| **Contradictory** | Can the model identify contradictions? | "The patient is 25 and has been practicing for 30 years" |

**Why this matters**: Gemini 3.1 Pro scores 97.3% on INSUFFICIENT but only 19.2% on UNDERDETERMINED — a 78 percentage-point gap that binary classification would completely hide.

## Quick Start

```python
from prometheus_ebm import Dataset, Evaluator, ECIScorer, EvalConfig

# Load the built-in 200-problem dataset
dataset = Dataset.load_default()
print(dataset.stats())

# Configure and run evaluation
config = EvalConfig(api_key="your-openrouter-key")
evaluator = Evaluator(config)

results = evaluator.run(
    problems=dataset.problems,
    models=["anthropic/claude-sonnet-4-5"],
)

# Score and analyze
for model, scored in results.items():
    scorer = ECIScorer(scored)
    print(scorer.summary())
    print(scorer.per_class_accuracy())
    print(scorer.failure_taxonomy())
```

## 7-Metric Evaluation Suite

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| **ECI** | Epistemological Calibration Index | Composite metacognitive score (0-1, higher=better) |
| **SDA** | Solvability Detection Accuracy | Can the model classify problem solvability? |
| **CA** | Conditional Accuracy | When it answers, is it correct? |
| **RP** | Refusal Precision | When it refuses, is refusal appropriate? |
| **ECE** | Expected Calibration Error | Does confidence match accuracy? (lower=better) |
| **HSS** | Hallucination Severity Score | Confident errors on impossible problems (lower=better) |
| **HGI** | Hysteresis Gap Index | Internal consistency of judgments (lower=better) |

## Epoch-1 Results

Tested on 4 frontier models × 297 prompts each:

| Model | ECI ↑ | SDA | CA | RP | ECE ↓ | HSS ↓ | HGI ↓ |
|-------|-------|-----|----|----|-------|-------|-------|
| Claude Opus 4.6 | **0.802** | 0.862 | 0.787 | 0.699 | 0.217 | 0.107 | 0.065 |
| Claude Sonnet 4.5 | 0.774 | 0.811 | 0.787 | 0.665 | 0.250 | 0.114 | **0.059** |
| Qwen3-235B | 0.716 | 0.710 | 0.733 | **0.759** | 0.371 | 0.262 | 0.099 |
| Gemini 3.1 Pro | 0.695 | 0.747 | 0.573 | 0.692 | 0.347 | **0.087** | 0.064 |

## Model Behavioural Profiling

```python
from prometheus_ebm import ModelProfiler

profiler = ModelProfiler()
profile = profiler.profile(scored_results, "anthropic/claude-opus-4-6")
print(profile.summary())
```

Output:
```
MODEL PROFILE: anthropic/claude-opus-4-6
  Risk:        ASSERTIVE   — Refusal rate matches unsolvable rate
  Calibration: ALIGNED     — 3.7pp confidence gap (decent self-knowledge)
  Stability:   FRAGILE     — Drops 9pp under decision stress
  Blindspot:   financial domain (60.3%), UNDERDETERMINED class (71.2%)
  Hallucination: MODERATE  — 10.7% on impossible problems
```

## Installation

```bash
pip install prometheus-ebm
```

Or from source:
```bash
git clone https://github.com/prometheus-ebm/prometheus-ebm.git
cd prometheus-ebm
pip install -e .
```

## Citation

```bibtex
@software{prometheus_ebm_2026,
  title={PROMETHEUS-EBM: Epistemological Benchmark for Metacognition},
  author={PROMETHEUS-EBM Team},
  year={2026},
  url={https://github.com/prometheus-ebm/prometheus-ebm}
}
```

## License

MIT License
