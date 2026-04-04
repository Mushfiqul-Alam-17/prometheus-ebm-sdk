# Independent Lab Run Guide (V5 Parity)

This guide shows how external labs can run PROMETHEUS-EBM with SDK behavior aligned to the `Final_V5.ipynb` protocol.

Repository:
- https://github.com/Mushfiqul-Alam-17/prometheus-ebm

SDK folder in repo:
- https://github.com/Mushfiqul-Alam-17/prometheus-ebm/tree/master/prometheus-ebm-sdk

## 1) Install from GitHub

```bash
git clone https://github.com/Mushfiqul-Alam-17/prometheus-ebm.git
cd prometheus-ebm/prometheus-ebm-sdk
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

## 2) Configure API route

Use one of:
- `provider="kaggle"` for Kaggle-hosted runs
- `provider="openrouter"`, `provider="openai"`, `provider="anthropic"` for direct providers
- `provider="custom"` for OpenAI-compatible lab gateways (requires `api_base_url`)

## 3) Run Multi-model (STANDARD or EXTENDED)

```python
import os
from prometheus_ebm import build_v5_config, run_v5_workflow

config = build_v5_config(
    mode="extended",  # or "standard"
    models=[
        "anthropic/claude-opus-4-6@default",
        "anthropic/claude-sonnet-4-6@default",
        "google/gemini-3.1-pro-preview",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-r1-0528",
    ],
    provider="custom",
    api_key=os.getenv("LAB_API_KEY"),
    api_base_url="https://your-lab-gateway.example/v1",
    output_dir="lab_out_extended",
    run_multistage=True,
    run_probes=True,
    run_statistics=True,
    run_research_grade_blocks=True,
)

results = run_v5_workflow(
    config,
    export_bundle=True,
    export_path="lab_out_extended/prometheus_sdk_v5_bundle.zip",
)

print(results.summary)
print(results.validate_research_grade())
```

## 4) Run Solo Deep Probe

```python
import os
from prometheus_ebm import build_v5_config, run_v5_workflow

config = build_v5_config(
    mode="deep_probe",
    models=["anthropic/claude-opus-4-6@default"],
    provider="custom",
    api_key=os.getenv("LAB_API_KEY"),
    api_base_url="https://your-lab-gateway.example/v1",
    output_dir="lab_out_deep_probe",
    run_multistage=True,
    run_probes=True,
    run_statistics=True,
    run_research_grade_blocks=True,
)

results = run_v5_workflow(
    config,
    export_bundle=True,
    export_path="lab_out_deep_probe/prometheus_sdk_v5_bundle.zip",
)

print(results.summary)
print(results.validate_research_grade())
```

## 5) What artifacts to expect

- `Final_Output_main.csv`
- `Final_Output_main.json`
- `run_profile.json`
- Epoch-1 exports (`prometheus_model_comparison.*`, `prometheus_item_level_results.*`)
- Epoch-2 exports (`probe_results.csv`, `multistage_results.csv`)
- Research-grade gate/card artifacts (when enabled)
- `prometheus_sdk_v5_bundle.zip`

## 6) Reproducibility checklist

- Pin to the same git commit/tag.
- Use the same dataset/probe files.
- Use the same model IDs (prefer pinned provider versions).
- Use the same mode, seeds, stress ratios, bootstrap rounds, and retry settings.
- Compare `run_profile.json` first, then final output tables.

## 7) Why results can still differ slightly across labs

The SDK logic and artifact contract are aligned, but remote model providers can update model backends over time. This can produce small output/score differences across dates or endpoints even with identical SDK settings.
