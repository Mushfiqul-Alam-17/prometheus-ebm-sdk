"""
PROMETHEUS-EBM Living Benchmark Generator
==========================================

Anti-contamination architecture for frontier AI evaluation.

The core insight: the TAXONOMY (4 solvability classes) and SCORING (ECI formula)
are permanent. Only the PROBLEM CONTENT rotates. This makes the benchmark
impossible to game through memorization while keeping scores comparable across epochs.

Three-tier anti-contamination system:
  Tier 1 — Epoch Versioning: Generate fresh problem sets quarterly
  Tier 2 — Parameterized Templates: Same epistemic structure, randomized surface values
  Tier 3 — Live Generation: Problems generated on-the-fly via LLM API

Quick Start:
    from prometheus_ebm.generator import EpochGenerator

    gen = EpochGenerator(epoch_id="v3", seed=42)
    problems = gen.generate(n_problems=200)
    gen.save(problems, output_dir="./epoch_v3")
"""

from prometheus_ebm.generator.engine import (
    EpochGenerator,
    generate_epoch,
    create_epoch_manifest,
    resolve_param,
    instantiate_template,
)

__all__ = [
    "EpochGenerator",
    "generate_epoch",
    "create_epoch_manifest",
    "resolve_param",
    "instantiate_template",
]
