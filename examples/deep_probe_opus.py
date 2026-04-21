"""
Example: Deep Probe a single model with 1,000 items.

This example runs an in-depth diagnostic evaluation on a single model
using an expanded dataset for maximum statistical power.
"""
from prometheus_ebm import build_v5_config, run_v5_workflow

# Configure a deep probe run on Claude Opus 4.6
config = build_v5_config(
    mode="deep_probe",
    models=["anthropic/claude-opus-4-6@default"],
    
    # Use Anthropic API directly for maximum control
    provider="anthropic",
    api_key="sk-ant-YOUR_KEY_HERE",  # Replace with your API key
    
    # Extended dataset for deep analysis
    n_items=1000,                 # 1,000 items for maximum signal
    stress_decision_ratio=0.30,   # Slightly higher stress for deep probe
    stress_clarity_ratio=0.15,
    
    # Statistical configuration
    seeds=[
        "prometheus-2026-deep-s1",
        "prometheus-2026-deep-s2",
        "prometheus-2026-deep-s3",  # More seeds for tighter CIs
    ],
    probe_seeds=[
        "prometheus-2026-deep-p1",
        "prometheus-2026-deep-p2",
        "prometheus-2026-deep-p3",
    ],
    bootstrap_iterations=5000,     # More iterations for precision
    
    # V5 Parity specific
    run_independent_judge_sensitivity=True,
    run_research_grade_blocks=True,
    
    # Runtime
    timeout_per_model=36000,       # 10 hours (single model, no rush)
    total_time_budget=43200,
)

print(config.summary())

# Run the deep probe
# results = run_v5_workflow(
#    config, 
#    export_bundle=True, 
#    export_path="opus_deep_probe_bundle.zip"
# )

print("\n[Note: Uncomment the run_v5_workflow lines above to execute the deep probe]")
print("[Warning: A 1,000-item probe costs approximately $150-$300 in API fees]")
