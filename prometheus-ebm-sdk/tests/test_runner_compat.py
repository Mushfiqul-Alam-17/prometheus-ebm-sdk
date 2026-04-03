import json
from pathlib import Path

from prometheus_ebm import PrometheusRunner, RunConfig


class _FakeProvider:
    def prompt(self, model_name: str, system: str, user: str) -> str:
        return (
            "FINAL_ANSWER: correctanswer\n"
            "SOLVABILITY_CLASS: Determinate\n"
            "CONFIDENCE: 80\n"
            "JUSTIFICATION_TYPE: Direct\n"
            "REASONING: Synthetic deterministic response for tests."
        )


def _make_dataset(path: Path, n_items: int = 10) -> None:
    rows = []
    for i in range(n_items):
        rows.append(
            {
                "problem_id": f"P-{i}",
                "domain": "mathematics",
                "question": f"What is item {i}?",
                "user": f"What is item {i}?",
                "ground_truth_answer": "correctanswer",
                "problem_class": "DETERMINATE",
                "correct_solvability_class": "Determinate",
                "rigor_mode": "base",
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_run_all_alias_and_export_zip_shorthand(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    _make_dataset(dataset_path, n_items=10)

    config = RunConfig(
        mode="standard",
        models=["llama-3.1-70b-versatile"],
        provider="openai",
        api_key="dummy-key",
        api_base_url="https://api.groq.com/openai/v1",
        n_items=10,
        dataset_path=str(dataset_path),
        run_probes=False,
        run_multistage=False,
        run_statistics=False,
        verbose=False,
        output_dir=str(tmp_path / "out"),
    )

    runner = PrometheusRunner(config=config, provider=_FakeProvider())
    results = runner.run_all()

    assert results.summary["n_models"] == 1
    assert results.summary["best_model"] == "llama-3.1-70b-versatile"

    results.export("zip")
    assert (tmp_path / "out" / "prometheus_sdk_v5_bundle.zip").exists()


def test_multistage_is_separate_from_epoch1_scoring(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    _make_dataset(dataset_path, n_items=10)

    config = RunConfig(
        mode="standard",
        models=["llama-3.1-70b-versatile"],
        provider="openai",
        api_key="dummy-key",
        api_base_url="https://api.groq.com/openai/v1",
        n_items=10,
        dataset_path=str(dataset_path),
        stress_decision_ratio=0.0,
        stress_clarity_ratio=0.0,
        run_probes=False,
        run_multistage=True,
        run_statistics=False,
        multistage_sample_n=4,
        verbose=False,
    )

    runner = PrometheusRunner(config=config, provider=_FakeProvider())
    results = runner.run()

    assert len(results.raw_dataframe) == 10
    assert "stage_a" not in results.raw_dataframe.columns
    assert len(results.multistage_dataframe) == config.multistage_sample_n
    assert "stage_a" in results.multistage_dataframe.columns
