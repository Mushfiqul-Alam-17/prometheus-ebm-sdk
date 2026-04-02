"""
cli.py — Command-line interface for PROMETHEUS-EBM

Usage:
    prometheus-ebm run --model anthropic/claude-sonnet-4-5 --api-key sk-...
    prometheus-ebm stats
    prometheus-ebm profile --results results.json
"""

import argparse
import json
import sys
import os

def cmd_stats(args):
    """Show dataset statistics."""
    from prometheus_ebm.dataset import Dataset
    ds = Dataset.load_default()
    stats = ds.stats()
    print(f"PROMETHEUS-EBM Dataset")
    print(f"  Total problems: {stats['total']}")
    print(f"  By class:")
    for cls, count in sorted(stats['by_class'].items()):
        print(f"    {cls:<18}: {count}")
    print(f"  By domain:")
    for dom, count in sorted(stats['by_domain'].items()):
        print(f"    {dom:<18}: {count}")


def cmd_run(args):
    """Run evaluation on one or more models."""
    from prometheus_ebm.dataset import Dataset
    from prometheus_ebm.evaluator import Evaluator, EvalConfig
    from prometheus_ebm.scorer import ECIScorer

    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        print("ERROR: Provide --api-key or set OPENROUTER_API_KEY env var")
        sys.exit(1)

    ds = Dataset.load_default()
    config = EvalConfig(
        api_key=api_key,
        stress_augment=not args.no_stress,
    )
    evaluator = Evaluator(config)

    models = args.model
    print(f"Running PROMETHEUS-EBM on {len(models)} model(s)...")
    print(f"  Problems: {len(ds)}")
    print(f"  Stress augmentation: {'off' if args.no_stress else 'on'}")

    def progress(model, i, n):
        if i % 10 == 0 or i == n:
            print(f"  [{model}] {i}/{n}")

    results = evaluator.run(ds.problems, models, progress_callback=progress)

    # Print results
    print(f"\n{'='*60}")
    for model, scored in results.items():
        scorer = ECIScorer(scored)
        print(f"\n{scorer.summary()}")
        print(f"\nPer-class accuracy:")
        for cls, data in scorer.per_class_accuracy().items():
            print(f"  {cls:<18}: {data['accuracy']:.1%} [{data['ci_lower']:.1%}, {data['ci_upper']:.1%}]")

    # Save results
    if args.output:
        comparison = evaluator.compare(results)
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def cmd_version(args):
    """Print version."""
    from prometheus_ebm import __version__
    print(f"prometheus-ebm {__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog='prometheus-ebm',
        description='PROMETHEUS-EBM: Epistemological Benchmark for Metacognition',
    )
    subparsers = parser.add_subparsers(dest='command')

    # stats
    sub_stats = subparsers.add_parser('stats', help='Show dataset statistics')
    sub_stats.set_defaults(func=cmd_stats)

    # run
    sub_run = subparsers.add_parser('run', help='Run evaluation')
    sub_run.add_argument('--model', nargs='+', required=True, help='Model(s) to evaluate')
    sub_run.add_argument('--api-key', help='OpenRouter API key')
    sub_run.add_argument('--no-stress', action='store_true', help='Disable stress augmentation')
    sub_run.add_argument('--output', '-o', help='Output JSON file for results')
    sub_run.set_defaults(func=cmd_run)

    # version
    sub_ver = subparsers.add_parser('version', help='Print version')
    sub_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
