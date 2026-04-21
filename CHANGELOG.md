# Changelog

All notable changes to **prometheus-ebm** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com).

---

## [1.1.0] — 2026-04

### Added
- `KAGGLE_MODEL_CATALOG` — canonical 34-model 1-based index list, mirroring
  `Final_V5.ipynb` Cell 4. Exported from the top-level `prometheus_ebm` package.
- `resolve_models_from_indices(indices, catalog=None)` — resolve model strings
  from 1-based integer indices. Enables notebook-style model selection (`[25, 5, 7]`)
  inside SDK scripts.
- `notebooks/prometheus_ebm_portable.ipynb` — fully self-contained notebook that
  runs on **Kaggle, Google Colab, JupyterLab, and any local Python environment**
  without any code changes. Platform detection happens automatically in C03.

### Changed
- `RunConfig.independent_judge_candidates` now defaults to `[]` and is
  auto-populated from `self.models` inside `apply_mode_defaults()`.
  This matches the V5 notebook fix:
  `INDEPENDENT_JUDGE_CANDIDATES = list(TARGET_MODELS)`.
  Pass an explicit non-empty list only when you need a custom judge pool.
- Version bumped to `1.1.0` in `pyproject.toml` and `__init__.py`.

### Fixed
- Judge pool was previously hardcoded to 5 specific model strings in `RunConfig`,
  causing a mismatch whenever users changed their evaluation models. The pool now
  always inherits from the evaluation target list.

---

## [1.0.1] — 2026-04

### Added
- Initial public release.
- `RunConfig` with `standard`, `extended`, `deep_probe` modes.
- `PrometheusRunner` with checkpoint resume, multi-stage protocol, and
  research-grade artifact generation.
- `ECIScorer`, `BrierDecomposition`, `Type2DPrime`, `HGIScorer`.
- Provider adapters: Kaggle, OpenRouter, OpenAI, Anthropic.
- `workflow_v5.build_v5_config()` and `workflow_v5.run_v5_workflow()`.
