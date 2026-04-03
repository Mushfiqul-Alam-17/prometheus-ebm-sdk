"""
Run configuration for PROMETHEUS-EBM benchmarking.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class RunConfig:
    """Configuration for a PROMETHEUS-EBM benchmark run.
    
    Args:
        mode: "compare" for multi-model leaderboard, "deep_probe" for single-model diagnostic
        models: List of model identifiers to evaluate
        provider: API provider - "kaggle", "openrouter", "anthropic", "openai", or "custom"
        api_key: API key for the chosen provider (not needed for "kaggle")
        api_base_url: Custom API base URL (for "custom" provider or OpenRouter)
        n_items: Number of base problems to use (200 = standard, 1000 = deep probe)
        stress_decision_ratio: Fraction of problems to apply decision stress augmentation
        stress_clarity_ratio: Fraction of problems to apply clarity stress augmentation
        seeds: List of reproducibility seeds for bootstrap resampling
        bootstrap_iterations: Number of bootstrap iterations for confidence intervals
        timeout_per_model: Maximum seconds per model before skipping
        total_time_budget: Total wall-clock seconds for the entire run
        time_reserve: Seconds reserved at end for analysis and export
        checkpoint_dir: Directory for per-model checkpoint files
        resume_from_checkpoint: Whether to load previous partial results
        output_dir: Directory for all output artifacts
        run_probes: Whether to run Epoch-2 adversarial probes
        run_multistage: Whether to run multi-stage adversarial protocol
        run_statistics: Whether to run bootstrap CI and significance tests
        verbose: Print progress during execution
    """
    # ── Core Mode ──────────────────────────────────────────────────────────────
    mode: Literal["compare", "deep_probe"] = "compare"
    models: List[str] = field(default_factory=lambda: [])
    
    # ── Provider ───────────────────────────────────────────────────────────────
    provider: Literal["kaggle", "openrouter", "anthropic", "openai", "custom"] = "kaggle"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    
    # ── Dataset ────────────────────────────────────────────────────────────────
    n_items: int = 200
    dataset_path: Optional[str] = None  # Custom dataset path
    stress_decision_ratio: float = 0.25
    stress_clarity_ratio: float = 0.10
    
    # ── Statistical ────────────────────────────────────────────────────────────
    seeds: List[str] = field(default_factory=lambda: ["prometheus-2026-s1", "prometheus-2026-s2"])
    bootstrap_iterations: int = 2000
    
    # ── Time Budget ────────────────────────────────────────────────────────────
    timeout_per_model: int = 10800       # 3 hours
    total_time_budget: int = 43200       # 12 hours
    time_reserve: int = 3600             # 1 hour for analysis
    
    # ── Checkpointing ──────────────────────────────────────────────────────────
    checkpoint_dir: str = "prometheus_checkpoints"
    resume_from_checkpoint: bool = True
    
    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "prometheus_output"
    
    # ── Feature Flags ──────────────────────────────────────────────────────────
    run_probes: bool = True
    run_multistage: bool = True
    run_statistics: bool = True
    verbose: bool = True
    
    def validate(self):
        """Validate configuration before running."""
        errors = []
        
        if not self.models:
            errors.append("At least one model must be specified in 'models'")
        
        if self.mode == "deep_probe" and len(self.models) > 1:
            errors.append("deep_probe mode supports only 1 model (for focused evaluation)")
        
        if self.provider != "kaggle" and not self.api_key:
            errors.append(f"API key required for provider '{self.provider}'")
        
        if self.n_items < 10:
            errors.append("n_items must be at least 10")
        
        if self.n_items > 200 and self.mode == "compare" and len(self.models) > 3:
            import warnings
            warnings.warn(
                f"Running {len(self.models)} models on {self.n_items} items may exceed time budget. "
                f"Consider using mode='deep_probe' for single-model evaluation with large datasets."
            )
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def summary(self) -> str:
        """Human-readable configuration summary."""
        lines = [
            f"=== PROMETHEUS-EBM Configuration ===",
            f"  Mode: {self.mode}",
            f"  Models: {', '.join(self.models)}",
            f"  Provider: {self.provider}",
            f"  Dataset: {self.n_items} base items + stress augmentation",
            f"  Stress: {self.stress_decision_ratio*100:.0f}% decision / {self.stress_clarity_ratio*100:.0f}% clarity",
            f"  Bootstrap: {self.bootstrap_iterations} iterations × {len(self.seeds)} seeds",
            f"  Timeout: {self.timeout_per_model//3600}h per model / {self.total_time_budget//3600}h total",
        ]
        return "\n".join(lines)
