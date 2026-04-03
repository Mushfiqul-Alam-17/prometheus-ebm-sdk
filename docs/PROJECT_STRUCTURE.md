# AG Project Structure

## Core Components

- `notebooks/`
  - Contains benchmark notebooks.
  - `benchmark-v4.ipynb` is the primary notebook for research-grade v1.

- `data/`
  - `prometheus_real_dataset.json`: Epoch-1 base dataset.
  - `probe_ambiguity.json`, `probe_contradictions.json`: Epoch-2 probe datasets.
  - `results/`: previously generated outputs and analysis snapshots.

- `prometheus-ebm/`
  - Python SDK source package (`prometheus_ebm/`).
  - Includes evaluator, scorer, stats, taxonomy, stress logic, and CLI.

- `tools/`
  - Utility scripts for notebook migration and results inspection.
  - `legacy_analysis/`: historical analysis scripts.
  - `maintenance/`: cleanup and packaging scripts.

- `kaggle/`
  - `input_bundle/` contains the exact JSON files to upload as Kaggle input.

- `outputs/`
  - Clean target directories for local output organization:
    - `outputs/epoch1/`
    - `outputs/epoch2/`
    - `outputs/research_grade/`

- `docs/`
  - Documentation and archived notes.

## Cleanup Applied

- Removed generated SDK build/cache directories:
  - `prometheus-ebm/prometheus_ebm/__pycache__/`
  - `prometheus-ebm/prometheus_ebm.egg-info/`
- Moved story note files to `docs/archive/`.
- Added Kaggle-ready input bundle under `kaggle/input_bundle/`.

## Compatibility Notes

- Notebook and SDK source paths were preserved.
- No benchmark notebook logic was removed.
- Existing historical results in `data/results/` were kept intact.
