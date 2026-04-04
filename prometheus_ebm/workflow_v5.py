"""Notebook V5 workflow helpers for portable SDK execution.

These helpers expose a thin, mode-aware wrapper around RunConfig + PrometheusRunner
so external labs can run the same workflow contract without depending on notebooks.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .config import RunConfig
from .runner import BenchmarkResults, PrometheusRunner


def build_v5_config(
    *,
    mode: str,
    models: List[str],
    provider: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    output_dir: str = "outputs",
    run_research_grade_blocks: bool = True,
    run_multistage: bool = True,
    run_probes: bool = True,
    verbose: bool = True,
    **overrides,
) -> RunConfig:
    """Create a notebook-parity RunConfig for standard/extended/deep_probe flows."""
    cfg = RunConfig(
        mode=mode,
        models=models,
        provider=provider,
        api_key=api_key,
        api_base_url=api_base_url,
        output_dir=output_dir,
        run_research_grade_blocks=run_research_grade_blocks,
        run_multistage=run_multistage,
        run_probes=run_probes,
        verbose=verbose,
        **overrides,
    )
    cfg.validate()
    return cfg


def run_v5_workflow(
    config: RunConfig,
    *,
    export_bundle: bool = True,
    export_path: Optional[str] = None,
) -> BenchmarkResults:
    """Run the full V5-equivalent SDK pipeline and optionally export the artifact bundle."""
    runner = PrometheusRunner(config)
    results = runner.run()

    if export_bundle:
        out_path = export_path or os.path.join(config.output_dir, "prometheus_sdk_v5_bundle.zip")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        results.export(out_path, "zip")

    return results
