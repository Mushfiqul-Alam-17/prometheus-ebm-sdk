"""Minimal local dry run against Groq via OpenAI-compatible API.

Usage (PowerShell):
    $env:OPENAI_API_KEY="gsk_..."
    python examples/test_groq.py
"""

import os

from prometheus_ebm import build_v5_config, run_v5_workflow


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    config = build_v5_config(
        mode="standard",
        models=["llama-3.3-70b-versatile"],
        provider="custom",
        api_key=api_key,
        api_base_url="https://api.groq.com/openai/v1",
        n_items=10,
        run_probes=True,
        run_multistage=False,
        run_statistics=True,
        run_research_grade_blocks=True,
        run_independent_judge_sensitivity=True,
        verbose=True,
    )

    results = run_v5_workflow(
        config,
        export_bundle=True,
        export_path="groq_test_bundle.zip"
    )

    print("Run complete.")
    print(f"Best model: {results.summary.get('best_model', 'N/A')}")
    print(f"Overall ECI: {results.summary.get('overall_eci', 'N/A')}")


if __name__ == "__main__":
    main()
