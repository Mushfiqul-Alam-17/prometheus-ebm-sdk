# PROMETHEUS-EBM SDK

[![PyPI](https://img.shields.io/pypi/v/prometheus-ebm.svg)](https://pypi.org/project/prometheus-ebm/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![Website](https://img.shields.io/badge/website-prometheusebm.com-blueviolet)](https://www.prometheusebm.com)

**The Living Benchmark for Epistemic Metacognition in Frontier AI Models**

**Author:** Mushfiqul Alam (Independent AI Researcher)
**Website:** [www.prometheusebm.com](https://www.prometheusebm.com) — Live leaderboard, benchmark dashboard, and model comparison results.

PROMETHEUS-EBM evaluates whether frontier AI models can recognize the *limits of their own knowledge* — not just answer questions, but understand when a question is unanswerable, ambiguous, or self-contradictory.

> **What's new in v1.1.0:** Added `KAGGLE_MODEL_CATALOG` (34-model 1-based index) + `resolve_models_from_indices()` for V5-style model selection. Fixed judge pool inheritance (`INDEPENDENT_JUDGE_CANDIDATES` now auto-populates from evaluation models). Added `notebooks/prometheus_ebm_portable.ipynb` — runs unchanged on Kaggle, Google Colab, JupyterLab, and local environments. Visit **[www.prometheusebm.com](https://www.prometheusebm.com)** for live benchmark results.
>
> **v1.0.1 (first stable release):** Complete Living Benchmark anti-contamination architecture · Full V5 adversarial evaluation protocol (multi-stage A/B/C/D) · Research-grade statistical gates (Bootstrap CI, Permutation Tests, Contamination Audit, Judge Sensitivity Analysis) · Production CLI (`prometheus-ebm`) · 12 parameterized templates for generating fresh un-memorizable evaluation epochs — making PROMETHEUS-EBM the first benchmark that is fundamentally impossible to game through training data contamination.

---

## Why This Exists

Current benchmarks (MMLU, GPQA, HumanEval) test **what a model knows**.
PROMETHEUS-EBM tests **whether a model knows what it does not know**.

This is a critical safety property. A model deployed in medicine, law, or finance that confidently answers when it *should* refuse is more dangerous than one that gets fewer questions right but knows its boundaries.

### The Scale Validity Gap

Our research revealed a structural deficit in frontier models: **metacognitive performance does not scale monotonically with model capability.** A mid-tier model (Claude Sonnet 4.6, ECI=0.884) outperformed its flagship sibling (Claude Opus 4.6, ECI=0.869) on epistemic calibration — a result that challenges the assumption that bigger = better for safety-critical deployment.

---

## The Living Benchmark Architecture

Static benchmarks die the moment they appear in training data. PROMETHEUS-EBM solves this with a **3-tier anti-contamination system**:

```
                    +----------------------------------+
                    |     PERMANENT (Never Changes)    |
                    |  4-Class Taxonomy  |  ECI Scoring |
                    +----------------------------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
    +-------v-------+   +--------v--------+   +--------v--------+
    |   TIER 1      |   |    TIER 2       |   |    TIER 3       |
    | Epoch Version |   | Parameterized   |   | Live Generation |
    |   Quarterly   |   |   Templates     |   |  On-the-fly     |
    |  regeneration |   | Same structure  |   | via LLM API     |
    |               |   | New values      |   |                 |
    +---------------+   +-----------------+   +-----------------+
```

**The key insight:** Each template encodes an *epistemic structure* — not content. A CONTRADICTORY template always contains irreconcilable conflicts regardless of what numbers are plugged in. The ground-truth label is guaranteed by structure, making every generated epoch automatically labeled and evaluation-ready.

```python
from prometheus_ebm.generator import EpochGenerator

# Generate a fresh un-memorizable epoch
gen = EpochGenerator(epoch_id="2026Q3", seed=42)
problems = gen.generate(n_problems=1000)
manifest = gen.save(problems, output_dir="./epoch_2026Q3")

# Verify zero overlap with previous epoch
gen.verify_no_overlap(problems, "path/to/epoch_v1_dataset.json")
```

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
| **ECI** (Epistemological Calibration Index) | 0-1 | Composite metacognition score |
| **SDA** (Solvability Detection Accuracy) | 0-1 | Can the model classify the problem type? |
| **CA** (Conditional Accuracy) | 0-1 | When it commits to an answer, is it correct? |
| **RP** (Refusal Precision) | 0-1 | When it refuses, was refusal appropriate? |
| **ECE** (Expected Calibration Error) | 0-1 | Does stated confidence match actual accuracy? |
| **HGI** (Hysteresis Gap Index) | 0-1 | Internal inconsistency (lower = better) |
| **Brier Score** | 0-1 | Calibration quality decomposed into Reliability, Resolution, Uncertainty |
| **Type-2 D-Prime** | -inf to +inf | How well the model's confidence signal distinguishes correct from incorrect answers |

### ECI Composition

```
ECI = 0.30 x SDA  +  0.25 x CA  +  0.20 x RP  +  0.15 x (1 - ECE)  +  0.10 x (1 - HSS)
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
from prometheus_ebm import build_v5_config, run_v5_workflow

config = build_v5_config(
    mode="extended",
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

results = run_v5_workflow(config, export_bundle=True)
```

### Generate Anti-Contamination Epochs

```python
from prometheus_ebm.generator import EpochGenerator

# Generate a fresh epoch with deterministic seeding
gen = EpochGenerator(epoch_id="v3", seed=42)
problems = gen.generate(n_problems=1000)

# Save dataset + manifest
manifest = gen.save(problems, output_dir="./epoch_v3")

# Verify zero overlap with any previous epoch
result = gen.verify_no_overlap(problems, "path/to/previous_epoch.json")
assert result["verified_clean"], "Contamination detected!"
```

### CLI for CI/CD Pipelines

```bash
# Generate a fresh evaluation epoch
prometheus-ebm generate --epoch v3 --n 1000 --seed 42 --output ./epoch_v3

# Run benchmark evaluation
prometheus-ebm run --mode extended --models claude-opus,gemini-pro --provider kaggle

# Show configuration and available resources
prometheus-ebm info

# List available parameterized templates
prometheus-ebm templates
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
    n_items=1000,
)
```

### Test with Groq (OpenAI-Compatible)

The OpenAI adapter supports custom endpoints, so you can route calls to Groq.

```python
import os
from prometheus_ebm import OpenAIProvider, PrometheusRunner, RunConfig

api_key = os.getenv("OPENAI_API_KEY")
provider = OpenAIProvider(api_key=api_key, base_url="https://api.groq.com/openai/v1")

config = RunConfig(
    mode="standard",
    models=["llama-3.1-70b-versatile"],
    provider="openai",
    api_key=api_key,
    api_base_url="https://api.groq.com/openai/v1",
    n_items=10,
    run_probes=True,
    run_multistage=False,
    run_statistics=True,
)

runner = PrometheusRunner(config=config, provider=provider)
results = runner.run_all()  # alias of run()
results.export("zip")
```

See `examples/test_groq.py` for a complete runnable example.

---

## Portable Notebook — Run Anywhere

> `notebooks/prometheus_ebm_portable.ipynb`

This is a **single-file, zero-setup** notebook that runs on every major platform with no code changes.

```
Platform      →  Evaluation Path
─────────────────────────────────────────────────────
Kaggle kernel →  uses kbench.evaluate() (primary)
Google Colab  →  uses direct API loop (api mode)
JupyterLab    →  uses direct API loop (api mode)
Local .py     →  use SDK: pip install prometheus-ebm
```

**Key design principles:**
- C03 auto-detects the runtime platform and installs `prometheus-ebm`.
- C08 defines `prometheus_ebm_task()` as a plain callable. On Kaggle, also registers it as a `@kbench.task`.
- C09 picks the right evaluation path automatically: kbench loop vs. direct API loop.
- All cells from C10 onwards (scoring, visualization, Epoch-2, multi-stage, export) are **identical** to `Final_V5.ipynb`.

**To run:**
1. Download `notebooks/prometheus_ebm_portable.ipynb`.
2. Place your dataset JSON files in the same directory (or mount them on Colab/Kaggle).
3. In C04: set `BENCHMARK_MODE`, `EXECUTION_MODE`, and your API key if running outside Kaggle.
4. Run all cells.

---

## Index-Based Model Selection (V5 Pattern)

Instead of hardcoding model strings, use 1-based catalog indices — exactly like the V5 notebook:

```python
from prometheus_ebm import (
    RunConfig, PrometheusRunner,
    KAGGLE_MODEL_CATALOG,
    resolve_models_from_indices,
)

# Same pattern as Final_V5 Cell 4:
# MULTI_MODEL_INDICES = [25, 5, 7, 11, 18]
models = resolve_models_from_indices([25, 5, 7, 11, 18])
print(models)
# [
#   'openai/gpt-5.4-2026-03-05',       # 25
#   'anthropic/claude-opus-4-7@default', # 5
#   'anthropic/claude-sonnet-4-6@default', # 7
#   'deepseek-ai/deepseek-v3.2',        # 11
#   'google/gemini-3.1-pro-preview',    # 18
# ]

config = RunConfig(
    mode='extended',
    provider='openrouter',
    api_key='sk-or-...',
    models=models,
)
runner = PrometheusRunner(config)
results = runner.run()
```

See `KAGGLE_MODEL_CATALOG` for the full 34-model indexed list.

### Use Your Own OpenAI-Compatible Endpoint

`provider="custom"` routes through the OpenAI adapter with your `api_base_url`, which is useful for local gateways, enterprise routers, and lab-hosted endpoints.

```python
from prometheus_ebm import RunConfig, PrometheusRunner

config = RunConfig(
    mode="standard",
    models=["your-lab-model"],
    provider="custom",
    api_key="sk-your-key",
    api_base_url="https://your.endpoint.example/v1",
)

results = PrometheusRunner(config).run()
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
hss = ECIScorer.compute_hss(answers_correct, true_classes, confidences)

eci = scorer.compute_eci(sda, ca, rp, ece, hss)

# Brier decomposition
brier = BrierDecomposition.compute(confidences, correctness)
# -> {'brier': 0.18, 'reliability': 0.03, 'resolution': 0.09, 'uncertainty': 0.24}

# D-Prime (metacognitive discrimination)
dprime = Type2DPrime.compute(confidences, correctness, threshold=0.7)
# -> {'d_prime': 1.24, 'hit_rate': 0.85, 'false_alarm_rate': 0.42}
```

---

## Supported Providers

| Provider | API Key Required | Models Available | Best For |
|----------|:---:|--------|----------|
| `kaggle` | No | 26 (Kaggle model pool) | Running inside Kaggle notebooks |
| `openrouter` | Yes | 100+ | Broadest model access with one key |
| `anthropic` | Yes | Claude family | Direct Anthropic API access |
| `openai` | Yes | GPT family | Direct OpenAI API access |
| `custom` | Yes | OpenAI-compatible endpoints | Self-hosted/lab APIs via custom base URL |

**Default behavior:** If no API key is provided, the SDK falls back to the Kaggle provider (which requires no authentication when running inside a Kaggle notebook).

---

## Dataset Navigation (V5 Lab Standard)

To ensure consistency with the `Final_V5.ipynb` research protocol, the SDK uses a tiered data structure:

- **Individual Model Testing** (`mode="deep_probe"`): Uses the **1,000-item Master Set** (`prometheus_1000_dataset.json`). Optimized for deep statistical significance on a single model.
- **Multi-model Comparison** (`mode="compare"`/`"standard"`): Uses the **200-item Leaderboard Subset** (`prometheus_200_multimodel_dataset.json`). Optimized for rapid benchmarking across multiple model families.
- **Living Benchmark** (`prometheus-ebm generate`): Uses the **Parameterized Template Engine** to produce fresh epochs on demand with zero content overlap.

The SDK automatically selects the appropriate file based on your `mode`, but you can always override this by providing a custom `dataset_path`.

---

## Anti-Contamination Generator Reference

### Python API

```python
from prometheus_ebm.generator import EpochGenerator

gen = EpochGenerator(epoch_id="2026Q3", seed=42)

# Generate problems
problems = gen.generate(n_problems=1000)  # List[Dict]

# Save dataset + manifest
manifest = gen.save(problems, output_dir="./output")

# Verify zero overlap
result = gen.verify_no_overlap(problems, "previous_epoch.json")
# -> {'overlap_count': 0, 'verified_clean': True, ...}

# List available templates
templates = EpochGenerator.list_templates()
# -> {'DETERMINATE': 5, 'CONTRADICTORY': 3, 'INSUFFICIENT': 2, 'UNDERDETERMINED': 2}
```

### CLI

```bash
# Full epoch generation
prometheus-ebm generate --epoch v4 --n 1000 --seed 42 --output ./epoch_v4

# Verify against previous epoch
prometheus-ebm generate --epoch v4 --n 200 --verify-against ./epoch_v3/prometheus_v3_dataset.json

# List templates
prometheus-ebm templates
```

### Template Structure

Each template encodes an **epistemic invariant** — the solvability class is determined by the problem's logical structure, not its surface values:

| Class | Structural Guarantee | Example |
|-------|---------------------|---------|
| DETERMINATE | All information for a unique answer is present | Financial net income calculation with all line items |
| CONTRADICTORY | Irreconcilable numerical/logical conflicts built-in | Revenue growth claim contradicted by actual figures |
| INSUFFICIENT | Data is analytically inadequate for the conclusion asked | Clinical trial with n=12 and no control group |
| UNDERDETERMINED | Multiple valid conclusions, no unique optimum | Startup valuation with three defensible term sheets |

---

## Configuration Reference

```python
RunConfig(
    # -- Mode --
    mode="standard",          # "standard", "extended", "deep_probe" ("compare" alias is still supported)
    models=[...],             # List of model identifiers

    # -- Provider --
    provider="kaggle",        # "kaggle", "openrouter", "anthropic", "openai", "custom"
    api_key=None,             # Required for non-Kaggle providers
    api_base_url=None,        # Required when provider="custom"

    # -- Dataset --
    n_items=200,              # Base problem count (200 standard, 1000 for deep probe)
    dataset_path=None,        # Path to custom dataset JSON (or None for bundled)
    stress_decision_ratio=0.25,  # Fraction with decision-pressure variants
    stress_clarity_ratio=0.10,   # Fraction with reduced-clarity variants

    # -- Statistical --
    seeds=["s1", "s2", "s3"],    # Epoch-1 resampling seeds
    probe_seeds=["p1", "p2", "p3"], # Epoch-2 resampling seeds
    bootstrap_iterations=3000,    # Bootstrap iterations for CIs
    pairwise_permutation_rounds=1000,
    multistage_sample_n=10,       # STANDARD/DEEP_PROBE default; EXTENDED uses 12
    multistage_model_strategy="top_bottom",  # "top_bottom", "all", "single_model"
    multistage_max_models=5,
    model_call_retries=1,
    judge_call_retries=0,

    # -- Time Budget --
    timeout_per_model=10800,  # Max seconds per model (default: 3h)
    total_time_budget=43200,  # Total budget (default: 12h)
    time_reserve=3600,        # Reserved for analysis (default: 1h)

    # -- Checkpointing --
    checkpoint_dir="prometheus_checkpoints",
    resume_from_checkpoint=True,

    # -- Output --
    output_dir="prometheus_output",
    final_output_basename="Final_Output_main",
    agi_metacog_target_score=0.85,

    # -- Feature Flags --
    run_probes=True,          # Epoch-2 adversarial probes
    run_multistage=True,      # Multi-stage belief revision protocol
    run_statistics=True,      # Bootstrap CIs and significance tests
    run_research_grade_blocks=True,
    run_independent_judge_sensitivity=False, # Optional (API-costly) criterion
    verbose=True,             # Print progress
)
```

---

## V5 Benchmark Results

Results from the PROMETHEUS-EBM v5.0 EXTENDED run (5 models x 324 items x 3 seeds):

### Epoch-1 Leaderboard

| Rank | Model | ECI | 95% CI | SDA |
|:---:|-------|:---:|--------|:---:|
| 1 | Claude Sonnet 4.6 | **0.884** | [0.878, 0.888] | 85.4% |
| 2 | Claude Opus 4.6 | 0.869 | [0.864, 0.877] | 84.3% |
| 3 | DeepSeek V3.2 | 0.815 | [0.800, 0.829] | 76.5% |
| 4 | DeepSeek R1-0528 | 0.785 | [0.774, 0.792] | 68.6% |
| 5 | Gemini 3.1 Pro | 0.767 | [0.745, 0.787] | 73.1% |

### Key Findings

1. **Sonnet beats Opus on ECI** (0.884 vs 0.869, statistically significant). The mid-tier model has better epistemic calibration than the top-tier model. Metacognition is not monotonic with scale.

2. **Opus leads on adversarial resilience.** Under the multi-stage protocol, Opus improved its accuracy by +13.9% after being challenged with counter-arguments. It correctly revised wrong answers without abandoning right ones.

3. **DeepSeek R1 classifies problems differently.** R1's solvability detection (SDA = 68.6%) diverges from all other models, and its evaluation perspective as a judge disagreed with peers at 16-20%. Chain-of-thought reasoning does not inherently improve metacognition.

4. **Gemini 3.1 Pro is the most overconfident.** Its stated confidence exceeds actual accuracy by 33 percentage points -- the largest gap in the benchmark.

---

## V5 Parity and Standalone Labs

The SDK export pipeline writes the same research-grade families used in `Final_V5.ipynb`, including:

- Epoch-1 bundle artifacts (`prometheus_item_level_results.*`, `prometheus_model_comparison.*`, `prometheus_results_export.zip`)
- Epoch-2 probe and multi-stage artifacts (`probe_results.csv`, `multistage_results.csv`, `prometheus_epoch2_export.zip`)
- RG artifacts (`rg_epoch1_*`, `rg_epoch2_*`, contamination audit, judge sensitivity report)
- Final gate/card artifacts (`research_grade_v1_gate.json`, `research_grade_v1_gate_criteria.csv`, `benchmark_card_research_grade_v1.md`)
- Master archive (`prometheus_FINAL_submission.zip`) included inside the exported zip

Minimal standalone flow for independent labs:

```python
from prometheus_ebm import PrometheusRunner, RunConfig

config = RunConfig(
    mode="extended",
    models=[
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4-6@default",
        "anthropic/claude-sonnet-4-6@default",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-r1-0528",
    ],
    provider="kaggle",
    run_multistage=True,
    run_research_grade_blocks=True,
)

runner = PrometheusRunner(config)
results = runner.run()
results.export("prometheus_sdk_v5_bundle.zip")
```

---

## Project Structure

```
prometheus-ebm-sdk/
├── prometheus_ebm/
│   ├── __init__.py          # Public API exports (v1.1.0)
│   ├── __main__.py          # CLI entrypoint
│   ├── config.py            # RunConfig, KAGGLE_MODEL_CATALOG, resolve_models_from_indices
│   ├── taxonomy.py          # 4-class solvability taxonomy
│   ├── scorer.py            # ECI, HGI, Brier, D-Prime, ReadinessScorer
│   ├── runner.py            # Benchmark orchestrator
│   ├── workflow_v5.py       # Notebook-parity helper entrypoints
│   ├── research_grade.py    # RG02-RG07 artifact pipeline
│   ├── generator/           # Living Benchmark Engine
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── templates.py
│   ├── data/                # Bundled datasets (1000- and 200-item)
│   └── providers/
│       ├── kaggle.py
│       ├── openrouter.py
│       ├── anthropic.py
│       └── openai.py
├── notebooks/
│   └── prometheus_ebm_portable.ipynb  # Portable notebook (Kaggle/Colab/local)
├── tests/
│   └── test_scorer.py
├── examples/
│   ├── compare_5_models.py
│   └── deep_probe_opus.py
├── CHANGELOG.md             # Version history
├── pyproject.toml
└── LICENSE
```

---

## Version History

| Version | Status | Highlights |
|---------|--------|-----------|
| **v1.1.0** | **Stable** | `KAGGLE_MODEL_CATALOG`, `resolve_models_from_indices()`, portable notebook (Kaggle/Colab/local), V5 judge-pool inheritance fix |
| **v1.0.1** | Stable | First major release. Living Benchmark engine, anti-contamination generator with 12 templates, CLI, full V5 adversarial protocol, research-grade statistical gates |

---

## License

MIT -- See [LICENSE](LICENSE) for details.

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
