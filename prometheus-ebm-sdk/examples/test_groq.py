"""Minimal local dry run against Groq via OpenAI-compatible API.

Usage (PowerShell):
    $env:OPENAI_API_KEY="gsk_..."
    python examples/test_groq.py
"""

import os

from prometheus_ebm import OpenAIProvider, PrometheusRunner, RunConfig


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    # Provider is OpenAI-compatible; base_url points to Groq.
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
        run_research_grade_blocks=True,
        verbose=True,
    )

    runner = PrometheusRunner(config=config, provider=provider)
    results = runner.run_all()
    results.export("zip")

    print("Run complete.")
    print(f"Best model: {results.summary['best_model']}")
    print(f"Overall ECI: {results.summary['overall_eci']}")


if __name__ == "__main__":
    main()
