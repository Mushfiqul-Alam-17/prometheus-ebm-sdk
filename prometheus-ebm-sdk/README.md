# PROMETHEUS-EBM SDK

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

**Benchmarking Epistemic Metacognition in AI Models**

PROMETHEUS-EBM evaluates whether frontier AI models can recognize the *limits of their own knowledge* — not just answer questions, but understand when a question is unanswerable, ambiguous, or self-contradictory.

> **Companion to the Kaggle notebook:** [PROMETHEUS-EBM v5.0](https://www.kaggle.com/code/mushfiqulalam007/final-bm-v4) — The full benchmark with live results from 5 frontier models.

---

## Why This Exists

Current benchmarks (MMLU, GPQA, HumanEval) test **what a model knows**.
PROMETHEUS-EBM tests **whether a model knows what it does not know**.

This is a critical safety property. A model deployed in medicine, law, or finance that confidently answers when it *should* refuse is more dangerous than one that gets fewer questions right but knows its boundaries.

---

## The 4-Class Solvability Taxonomy

Every problem is classified into one of four epistemic categories:

| Class | Description | Expected Model Behavior |
|-------|-------------|------------------------|
| **Determinate** | One clear answer exists | Answer confidently |
| **Underdetermined** | Multiple valid interpretations | Flag the ambiguity |
| **Insufficient** | Critical information is missing | Refuse to answer definitively |
| **Contradictory** | The premises conflict | Detect the contradiction |

Models are scored on whether they correctly identify *which category* a problem falls into — not just whether they produce the correct final answer.

---

## Scoring Framework

| Metric | Range | What It Measures |
|--------|-------|-----------------|
| **ECI** (Epistemological Calibration Index) | 0–1 | Composite metacognition score |
| **SDA** (Solvability Detection Accuracy) | 0–1 | Can the model classify the problem type? |
| **CA** (Conditional Accuracy) | 0–1 | When it commits to an answer, is it correct? |
| **RP** (Refusal Precision) | 0–1 | When it refuses, was refusal appropriate? |
| **ECE** (Expected Calibration Error) | 0–1 | Does stated confidence match actual accuracy? |
| **HGI** (Hysteresis Gap Index) | 0–1 | Internal inconsistency (lower = better) |
| **Brier Score** | 0–1 | Calibration quality decomposed into Reliability, Resolution, Uncertainty |
| **Type-2 D-Prime** | -∞ to +∞ | How well the model's confidence signal distinguishes correct from incorrect answers |

### ECI Composition

```
ECI = 0.30 × SDA  +  0.25 × CA  +  0.20 × RP  +  0.15 × (1 - ECE)  +  0.10 × (1 - HSS)
```

---

## Installation

```bash
pip install prometheus-ebm

# With specific provider support:
pip install "prometheus-ebm[anthropic]"   # For Claude API
pip install "prometheus-ebm[openai]"      # For OpenAI API
pip install "prometheus-ebm[all]"         # All providers
```

---

## Quick Start

### Compare Multiple Models

```python
from prometheus_ebm import PrometheusRunner, RunConfig

config = RunConfig(
    mode="compare",
    models=[
        "anthropic/claude-opus-4-6@default",
        "anthropic/claude-sonnet-4-6@default",
        "google/gemini-3.1-pro-preview",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-r1-0528",
    ],
    provider="kaggle",          # No API key needed
    n_items=200,                # Standard dataset (200 base problems)
    stress_decision_ratio=0.40, # EXTENDED mode stress
    stress_clarity_ratio=0.20,
)

runner = PrometheusRunner(config)
results = runner.run()
results.export("comparison.csv")
```

### Deep Probe a Single Model (1,000 Items)

```python
config = RunConfig(
    mode="deep_probe",
    models=["anthropic/claude-opus-4-6"],
    provider="anthropic",
    api_key="sk-ant-...",
    n_items=1000,
    stress_decision_ratio=0.30,
    bootstrap_iterations=3000,
)

runner = PrometheusRunner(config)
results = runner.run()
results.export("opus_deep_probe.csv")
```

### Use with OpenRouter (Access 100+ Models)

```python
config = RunConfig(
    mode="compare",
    models=["anthropic/claude-opus-4-6", "google/gemini-3.1-pro"],
    provider="openrouter",
    api_key="sk-or-...",
)
```

### Use with OpenAI

```python
config = RunConfig(
    mode="deep_probe",
    models=["gpt-5.4"],
    provider="openai",
    api_key="sk-...",
    n_items=200,
)
```

### Using Custom Datasets

The SDK comes bundled with 4 default datasets out of the box (the full 1,000-item deep probe, the 200-item standard, the ambiguity probe, and the contradiction probe). 

If you want to evaluate models on your own specialized dataset, format your test array as a JSON file matching the 4-class taxonomy, and pass the path directly to the `RunConfig`:

```python
config = RunConfig(
    mode="standard",
    models=["anthropic/claude-opus-4-6"],
    provider="anthropic",
    api_key="sk-...",
    dataset_path="c:/path/to/your/custom_dataset.json" # Overrides the defaults
)
```

### Scoring Only (Bring Your Own Data)

If you already have model responses and just need the ECI/Brier/D-Prime scores:

```python
from prometheus_ebm import ECIScorer, BrierDecomposition, Type2DPrime

scorer = ECIScorer()

# Compute individual components
sda = ECIScorer.compute_sda(predicted_classes, true_classes)
ca  = ECIScorer.compute_ca(answers_correct, true_classes)
rp  = ECIScorer.compute_rp(predicted_classes, true_classes)
ece = ECIScorer.compute_ece(confidences, correctness)
hss = ECIScorer.compute_hss(predicted_classes, true_classes, answers_given)

eci = scorer.compute_eci(sda, ca, rp, ece, hss)

# Brier decomposition
brier = BrierDecomposition.compute(confidences, correctness)
# → {'brier': 0.18, 'reliability': 0.03, 'resolution': 0.09, 'uncertainty': 0.24}

# D-Prime (metacognitive discrimination)
dprime = Type2DPrime.compute(confidences, correctness, threshold=0.7)
# → {'d_prime': 1.24, 'hit_rate': 0.85, 'false_alarm_rate': 0.42}
```

---

## Supported Providers

| Provider | API Key Required | Models Available | Best For |
|----------|:---:|--------|----------|
| `kaggle` | No | 26 (Kaggle model pool) | Running inside Kaggle notebooks |
| `openrouter` | Yes | 100+ | Broadest model access with one key |
| `anthropic` | Yes | Claude family | Direct Anthropic API access |
| `openai` | Yes | GPT family | Direct OpenAI API access |

**Default behavior:** If no API key is provided, the SDK falls back to the Kaggle provider (which requires no authentication when running inside a Kaggle notebook).

---

## Configuration Reference

```python
RunConfig(
    # ── Mode ──
    mode="compare",           # "compare" (multi-model) or "deep_probe" (single-model)
    models=[...],             # List of model identifiers

    # ── Provider ──
    provider="kaggle",        # "kaggle", "openrouter", "anthropic", "openai"
    api_key=None,             # Required for non-Kaggle providers
    api_base_url=None,        # Custom API endpoint (for self-hosted models)

    # ── Dataset ──
    n_items=200,              # Base problem count (200 standard, 1000 for deep probe)
    dataset_path=None,        # Path to custom dataset JSON (or None for bundled)
    stress_decision_ratio=0.25,  # Fraction with decision-pressure variants
    stress_clarity_ratio=0.10,   # Fraction with reduced-clarity variants

    # ── Statistical ──
    seeds=["s1", "s2"],       # Reproducibility seeds for bootstrap
    bootstrap_iterations=2000, # Bootstrap iterations for CIs

    # ── Time Budget ──
    timeout_per_model=10800,  # Max seconds per model (default: 3h)
    total_time_budget=43200,  # Total budget (default: 12h)
    time_reserve=3600,        # Reserved for analysis (default: 1h)

    # ── Checkpointing ──
    checkpoint_dir="prometheus_checkpoints",
    resume_from_checkpoint=True,

    # ── Output ──
    output_dir="prometheus_output",

    # ── Feature Flags ──
    run_probes=True,          # Epoch-2 adversarial probes
    run_multistage=True,      # Multi-stage belief revision protocol
    run_statistics=True,      # Bootstrap CIs and significance tests
    verbose=True,             # Print progress
)
```

---

## V5 Benchmark Results

Results from the PROMETHEUS-EBM v5.0 EXTENDED run (5 models × 324 items × 3 seeds):

### Epoch-1 Leaderboard

| Rank | Model | ECI | 95% CI | SDA |
|:---:|-------|:---:|--------|:---:|
| 🥇 | Claude Sonnet 4.6 | **0.884** | [0.878, 0.888] | 85.4% |
| 🥈 | Claude Opus 4.6 | 0.869 | [0.864, 0.877] | 84.3% |
| 🥉 | DeepSeek V3.2 | 0.815 | [0.800, 0.829] | 76.5% |
| 4 | DeepSeek R1-0528 | 0.785 | [0.774, 0.792] | 68.6% |
| 5 | Gemini 3.1 Pro | 0.767 | [0.745, 0.787] | 73.1% |

### Key Findings

1. **Sonnet beats Opus on ECI** (0.884 vs 0.869, statistically significant). The mid-tier model has better epistemic calibration than the top-tier model. Metacognition is not monotonic with scale.

2. **Opus leads on adversarial resilience.** Under the multi-stage protocol, Opus improved its accuracy by +13.9% after being challenged with counter-arguments. It correctly revised wrong answers without abandoning right ones.

3. **DeepSeek R1 classifies problems differently.** R1's solvability detection (SDA = 68.6%) diverges from all other models, and its evaluation perspective as a judge disagreed with peers at 16–20%. Chain-of-thought reasoning does not inherently improve metacognition.

4. **Gemini 3.1 Pro is the most overconfident.** Its stated confidence exceeds actual accuracy by 33 percentage points — the largest gap in the benchmark.

---

## Project Structure

```
prometheus-ebm-sdk/
├── prometheus_ebm/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # RunConfig dataclass
│   ├── taxonomy.py          # 4-class solvability taxonomy
│   ├── scorer.py            # ECI, HGI, Brier, D-Prime
│   ├── runner.py            # Benchmark orchestrator
│   ├── data/                # Bundled dataset (200 problems)
│   └── providers/
│       ├── kaggle.py        # Kaggle kbench adapter
│       ├── openrouter.py    # OpenRouter API adapter
│       ├── anthropic.py     # Anthropic Claude adapter
│       └── openai.py        # OpenAI adapter
├── tests/
│   └── test_scorer.py       # Unit tests for scoring engine
├── examples/
│   ├── compare_5_models.py  # Multi-model comparison example
│   └── deep_probe_opus.py   # Single-model deep probe example
├── pyproject.toml           # Package configuration
└── LICENSE
```

---

## Roadmap

| Version | Status | Features |
|---------|--------|----------|
| **v0.1.0** | ✅ Current | Scorer (ECI, Brier, D-Prime), Taxonomy, Config, Provider adapters |
| **v0.2.0** | Planned | Full evaluation loop, stress augmentation engine, export pipeline |
| **v0.3.0** | Planned | Bootstrap CI, pairwise significance, contamination audit |
| **v1.0.0** | Planned | 1,000-item dataset, CLI tool, HTML report generator |

---

## License

MIT — See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{alam2026prometheus,
  title   = {PROMETHEUS-EBM: Benchmarking Epistemic Metacognition in Frontier AI Models},
  author  = {Mushfiqul Alam},
  year    = {2026},
  url     = {https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk}
}
```
