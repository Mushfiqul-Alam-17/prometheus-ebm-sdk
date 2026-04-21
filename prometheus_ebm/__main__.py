"""
PROMETHEUS-EBM CLI
==================

Command-line interface for running PROMETHEUS-EBM evaluations and generating
anti-contamination benchmark epochs.

Usage:
    # Run benchmark evaluation
    prometheus-ebm run --mode extended --models claude-opus,gemini-pro --provider kaggle

    # Generate a fresh anti-contamination epoch
    prometheus-ebm generate --epoch v3 --n 1000 --seed 42 --output ./epoch_v3

    # Show current configuration and version
    prometheus-ebm info

    # List available templates
    prometheus-ebm templates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import prometheus_ebm


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the PROMETHEUS-EBM CLI."""
    parser = argparse.ArgumentParser(
        prog="prometheus-ebm",
        description=(
            "PROMETHEUS-EBM: Benchmarking Epistemic Metacognition in AI Models.\n"
            "The world's first living benchmark with anti-contamination architecture."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"prometheus-ebm {prometheus_ebm.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run a PROMETHEUS-EBM benchmark evaluation",
        description="Execute the full V5 evaluation pipeline on one or more models.",
    )
    run_parser.add_argument(
        "--mode", type=str, default="standard",
        choices=["standard", "extended", "compare", "deep_probe"],
        help="Benchmark mode (default: standard)",
    )
    run_parser.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated list of model identifiers to evaluate",
    )
    run_parser.add_argument(
        "--provider", type=str, default="kaggle",
        choices=["kaggle", "openrouter", "anthropic", "openai", "custom"],
        help="API provider (default: kaggle)",
    )
    run_parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the chosen provider",
    )
    run_parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a custom dataset JSON file",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default="prometheus_output",
        help="Directory for output artifacts (default: prometheus_output)",
    )
    run_parser.add_argument(
        "--n-items", type=int, default=None,
        help="Number of base problems (auto-selected by mode if not specified)",
    )
    run_parser.add_argument(
        "--no-probes", action="store_true",
        help="Skip Epoch-2 adversarial probes",
    )
    run_parser.add_argument(
        "--no-multistage", action="store_true",
        help="Skip multi-stage adversarial protocol",
    )
    run_parser.add_argument(
        "--export", action="store_true",
        help="Generate full export bundle after evaluation",
    )
    run_parser.add_argument(
        "--export-path", type=str, default=None,
        help="Custom path for export bundle (defaults to <output-dir>/prometheus_sdk_v5_bundle.zip)",
    )
    run_parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True,
        help="Print progress during execution (use --no-verbose to disable)",
    )

    # ── generate ──────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a fresh anti-contamination benchmark epoch",
        description=(
            "Generate a new epoch of problems using the Tier 2 parameterized template engine. "
            "Each epoch produces problems with identical epistemic structure but different "
            "surface content, making the benchmark impossible to game through memorization."
        ),
    )
    gen_parser.add_argument(
        "--epoch", type=str, default="v2",
        help="Epoch identifier (e.g., v2, v3, 2026Q2)",
    )
    gen_parser.add_argument(
        "--n", type=int, default=100,
        help="Number of problems to generate (default: 100)",
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: non-deterministic)",
    )
    gen_parser.add_argument(
        "--output", type=str, default=".",
        help="Output directory for dataset and manifest files",
    )
    gen_parser.add_argument(
        "--verify-against", type=str, default=None,
        help="Path to a previous epoch dataset to verify zero overlap",
    )
    gen_parser.add_argument(
        "--format", type=str, default="json",
        choices=["json"],
        help="Output format (default: json)",
    )

    # ── info ──────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "info",
        help="Show version, configuration, and available resources",
    )

    # ── templates ─────────────────────────────────────────────────────────
    subparsers.add_parser(
        "templates",
        help="List available parameterized templates",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a benchmark run."""
    from prometheus_ebm import PrometheusRunner, RunConfig

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: At least one model must be specified with --models")
        return 1

    config = RunConfig(
        mode=args.mode,
        models=models,
        provider=args.provider,
        api_key=args.api_key,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        run_probes=not args.no_probes,
        run_multistage=not args.no_multistage,
        verbose=args.verbose,
    )

    if args.n_items is not None:
        config.n_items = args.n_items

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error:\n{e}")
        return 1

    print(config.summary())
    print()

    runner = PrometheusRunner(config)
    results = runner.run()

    if args.export:
        export_path = args.export_path or str(Path(args.output_dir) / "prometheus_sdk_v5_bundle.zip")
        try:
            results.export(export_path, "zip")
            print(f"Export bundle written to: {export_path}")
        except Exception as e:
            print(f"Export error:\n{e}")
            return 1

    print(f"\nEvaluation complete. Results saved to: {args.output_dir}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a fresh anti-contamination epoch."""
    from prometheus_ebm.generator import EpochGenerator

    print(f"PROMETHEUS-EBM Living Benchmark Generator v{prometheus_ebm.__version__}")
    print(f"  Epoch: {args.epoch}")
    print(f"  Problems: {args.n}")
    print(f"  Seed: {args.seed or 'non-deterministic'}")
    print(f"  Output: {args.output}")
    print()

    gen = EpochGenerator(epoch_id=args.epoch, seed=args.seed)
    problems = gen.generate(n_problems=args.n, verbose=True)
    gen.save(problems, output_dir=args.output)

    if args.verify_against:
        print()
        result = gen.verify_no_overlap(problems, args.verify_against)
        if not result["verified_clean"]:
            print(f"\nWARNING: {result['overlap_count']} hash collisions detected!")
            return 1

    print(f"\n[OK] Generated {len(problems)} problems for epoch '{args.epoch}'")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Show version and resource information."""
    from prometheus_ebm.scorer import ECIScorer
    from prometheus_ebm.taxonomy import SolvabilityClass, BENCHMARK_DOMAINS

    print(f"PROMETHEUS-EBM SDK v{prometheus_ebm.__version__}")
    print(f"  PyPI: https://pypi.org/project/prometheus-ebm/")
    print(f"  GitHub: https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk")
    print()
    print("Epistemic Taxonomy:")
    for cls in SolvabilityClass:
        print(f"  {cls.value:20s}  {cls.description}")
    print()
    print(f"Benchmark Domains: {', '.join(BENCHMARK_DOMAINS)}")
    print()
    print("ECI Weights:")
    for component, weight in ECIScorer.WEIGHTS.items():
        print(f"  {component.upper():5s}: {weight:.0%}")
    print()
    print("Bundled Datasets:")
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        for f in sorted(data_dir.glob("*.json")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.0f} KB)")
    print()
    print("Generator Templates:")
    try:
        from prometheus_ebm.generator import EpochGenerator
        counts = EpochGenerator.list_templates()
        for cls, count in counts.items():
            print(f"  {cls:20s}: {count} templates")
    except Exception:
        print("  (generator module not available)")
    return 0


def _cmd_templates(_args: argparse.Namespace) -> int:
    """List available parameterized templates."""
    from prometheus_ebm.generator.templates import PARAMETERIZED_TEMPLATES

    print(f"PROMETHEUS-EBM Template Library")
    print(f"{'=' * 60}")
    for cls, templates in PARAMETERIZED_TEMPLATES.items():
        print(f"\n{cls} ({len(templates)} templates):")
        for t in templates:
            print(f"  [{t['id']}] {t['domain']}/{t['subtopic']}")
            # Show first 80 chars of the template
            preview = t["template"][:80].replace("\n", " ")
            print(f"    {preview}...")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "run": _cmd_run,
        "generate": _cmd_generate,
        "info": _cmd_info,
        "templates": _cmd_templates,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
