# AG Workspace

This workspace contains the PROMETHEUS benchmark notebooks, SDK, datasets, and tooling.

## Quick Navigation

- `notebooks/`
  - `benchmark-v4.ipynb` (current research-grade notebook)
  - `benchmark-v3.ipynb` (previous version)
- `data/`
  - `prometheus_200_multimodel_dataset.json`
  - `probe_ambiguity.json`
  - `probe_contradictions.json`
  - `results/` (legacy and exported artifacts)
- `prometheus-ebm/`
  - SDK package source (`prometheus_ebm/`)
  - package metadata (`setup.py`)
- `tools/`
  - notebook/build utilities
  - `legacy_analysis/` scripts
  - `maintenance/` cleanup and Kaggle prep scripts
- `kaggle/`
  - `input_bundle/` ready-to-upload Kaggle input dataset files
- `outputs/`
  - target folders for organized local outputs (`epoch1`, `epoch2`, `research_grade`)
- `docs/`
  - run guides, project map, archived notes
  - post-run dataset decision framework (`POSTRUN_DATASET_DECISION_PLAN.md`)

## Current Status (as of 2026-03-28)

- `benchmark-v4.ipynb` contains all research-grade v1 automation blocks (RG01-RG06).
- The notebook has not been executed yet in this workspace.
- No v4 research-grade artifacts have been generated locally yet.

## Recommended Run Entry Point

1. Read `docs/KAGGLE_RUN_GUIDE.md`.
2. Upload files from `kaggle/input_bundle/` as a Kaggle dataset.
3. Run `notebooks/benchmark-v4.ipynb` in Kaggle from top to bottom.
4. Confirm final gate outputs (`research_grade_v1_gate.json`, benchmark card, and criteria CSV).
