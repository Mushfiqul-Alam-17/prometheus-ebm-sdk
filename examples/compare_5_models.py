"""
Example: Compare multiple models using PROMETHEUS-EBM.

This example evaluates 5 frontier AI models on the full PROMETHEUS dataset
to produce a leaderboard ranking by epistemic metacognition quality.
"""
from prometheus_ebm import build_v5_config, run_v5_workflow

# Configure a multi-model comparison run
config = build_v5_config(
    mode="extended",
    models=[
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4-6@default",
        "anthropic/claude-sonnet-4-6@default",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-r1-0528",
    ],
    
    # Provider: "kaggle" (no API key), "openrouter", "anthropic", "openai"
    provider="kaggle",
    
    # Dataset configuration
    n_items=200,                  # 200 base items (standard benchmark)
    stress_decision_ratio=0.25,   # 25% of items get decision stress
    stress_clarity_ratio=0.10,    # 10% of items get clarity stress
    
    # Statistical configuration
    seeds=["prometheus-2026-s1", "prometheus-2026-s2", "prometheus-2026-s3"],
    probe_seeds=["prometheus-2026-p1", "prometheus-2026-p2", "prometheus-2026-p3"],
    bootstrap_iterations=3000,
    
    # V5 Parity specific
    run_independent_judge_sensitivity=True,
    run_research_grade_blocks=True,
    
    # Runtime safety
    timeout_per_model=10800,      # 3 hours per model
    total_time_budget=43200,      # 12 hours total
)

print(config.summary())

# Run the benchmark
# results = run_v5_workflow(
#    config, 
#    export_bundle=True, 
#    export_path="prometheus_sdk_v5_bundle.zip"
# )

print("\n[Note: Uncomment the run_v5_workflow lines above to execute the full benchmark]")
