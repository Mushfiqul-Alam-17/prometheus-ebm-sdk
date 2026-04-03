"""Run configuration for PROMETHEUS-EBM benchmarking."""

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
    # "compare" is kept as a compatibility alias of "standard".
    mode: Literal["standard", "extended", "compare", "deep_probe"] = "standard"
    models: List[str] = field(default_factory=lambda: [])
    
    # ── Provider ───────────────────────────────────────────────────────────────
    provider: Literal["kaggle", "openrouter", "anthropic", "openai", "custom"] = "kaggle"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    judge_model: Optional[str] = None
    
    # ── Dataset ────────────────────────────────────────────────────────────────
    n_items: int = 200
    dataset_path: Optional[str] = None  # Custom dataset path
    stress_decision_ratio: float = 0.25
    stress_clarity_ratio: float = 0.10
    
    # ── Statistical ────────────────────────────────────────────────────────────
    seeds: List[str] = field(default_factory=lambda: ["prometheus-2026-s1", "prometheus-2026-s2"])
    probe_seeds: List[str] = field(default_factory=list)
    bootstrap_iterations: int = 2000
    pairwise_permutation_rounds: int = 1000
    min_seeds_required: int = 2
    
    # ── Time Budget ────────────────────────────────────────────────────────────
    timeout_per_model: int = 10800       # 3 hours
    total_time_budget: int = 43200       # 12 hours
    time_reserve: int = 3600             # 1 hour for analysis
    
    # ── Checkpointing ──────────────────────────────────────────────────────────
    checkpoint_dir: str = "prometheus_checkpoints"
    resume_from_checkpoint: bool = True
    
    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "prometheus_output"
    master_bundle_name: str = "prometheus_FINAL_submission.zip"
    
    run_probes: bool = True
    run_multistage: bool = False
    run_statistics: bool = True
    run_research_grade_blocks: bool = True
    run_independent_judge_sensitivity: bool = False

    independent_judge_candidates: List[str] = field(default_factory=lambda: [
        "anthropic/claude-opus-4-6@default",
        "anthropic/claude-sonnet-4-6@default",
        "google/gemini-3.1-pro-preview",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-r1-0528",
    ])
    independent_judge_sample_max: int = 60
    judge_sensitivity_max_disagreement: float = 0.25
    multistage_sample_n: int = 10

    verbose: bool = True

    def normalized_mode(self) -> Literal["standard", "extended", "deep_probe"]:
        """Return canonical mode naming used by the V5 notebook contract."""
        return "standard" if self.mode == "compare" else self.mode  # type: ignore[return-value]

    def apply_mode_defaults(self):
        """Apply V5-compatible defaults while preserving explicit custom overrides."""
        mode = self.normalized_mode()

        if mode == "extended":
            self.min_seeds_required = max(self.min_seeds_required, 3)
            if self.stress_decision_ratio == 0.25:
                self.stress_decision_ratio = 0.40
            if self.stress_clarity_ratio == 0.10:
                self.stress_clarity_ratio = 0.20
            if self.bootstrap_iterations < 3000:
                self.bootstrap_iterations = 3000
            if self.timeout_per_model < 14400:
                self.timeout_per_model = 14400
            if self.multistage_sample_n < 12:
                self.multistage_sample_n = 12

        elif mode == "deep_probe":
            self.min_seeds_required = max(self.min_seeds_required, 3)
            if self.n_items < 1000 and self.dataset_path is None:
                self.n_items = 1000
            if self.stress_decision_ratio == 0.25:
                self.stress_decision_ratio = 0.30
            if self.stress_clarity_ratio == 0.10:
                self.stress_clarity_ratio = 0.15
            if self.timeout_per_model < 36000:
                self.timeout_per_model = 36000
            if self.multistage_sample_n < 10:
                self.multistage_sample_n = 10

        else:
            self.min_seeds_required = max(2, self.min_seeds_required)
            if self.multistage_sample_n < 10:
                self.multistage_sample_n = 10

        if not self.probe_seeds:
            # If not explicitly provided, align probe seeds with core seeds.
            self.probe_seeds = list(self.seeds)

        # Ensure seeds satisfy current minimum requirements.
        while len(self.seeds) < self.min_seeds_required:
            self.seeds.append(f"prometheus-2026-s{len(self.seeds) + 1}")
        while len(self.probe_seeds) < self.min_seeds_required:
            self.probe_seeds.append(f"prometheus-2026-p{len(self.probe_seeds) + 1}")
    
    def validate(self):
        """Validate configuration before running."""
        self.apply_mode_defaults()
        errors = []
        
        if not self.models:
            errors.append("At least one model must be specified in 'models'")
        
        if self.normalized_mode() == "deep_probe" and len(self.models) > 1:
            errors.append("deep_probe mode supports only 1 model (for focused evaluation)")
        
        if self.provider != "kaggle" and not self.api_key:
            errors.append(f"API key required for provider '{self.provider}'")
        
        if self.n_items < 10:
            errors.append("n_items must be at least 10")
        
        if self.n_items > 200 and self.normalized_mode() != "deep_probe" and len(self.models) > 3:
            import warnings
            warnings.warn(
                f"Running {len(self.models)} models on {self.n_items} items may exceed time budget. "
                f"Consider using mode='deep_probe' for single-model evaluation with large datasets."
            )

        if len(self.seeds) < self.min_seeds_required:
            errors.append(
                f"At least {self.min_seeds_required} seeds are required, got {len(self.seeds)}"
            )

        if len(self.probe_seeds) < self.min_seeds_required:
            errors.append(
                f"At least {self.min_seeds_required} probe seeds are required, got {len(self.probe_seeds)}"
            )
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def summary(self) -> str:
        """Human-readable configuration summary."""
        mode = self.normalized_mode()
        lines = [
            f"=== PROMETHEUS-EBM Configuration ===",
            f"  Mode: {mode}",
            f"  Models: {', '.join(self.models)}",
            f"  Provider: {self.provider}",
            f"  Dataset: {self.n_items} base items + stress augmentation",
            f"  Stress: {self.stress_decision_ratio*100:.0f}% decision / {self.stress_clarity_ratio*100:.0f}% clarity",
            f"  Bootstrap: {self.bootstrap_iterations} iterations × {len(self.seeds)} epoch seeds",
            f"  Probe Seeds: {len(self.probe_seeds)}",
            f"  Multi-stage sample n: {self.multistage_sample_n}",
            f"  Timeout: {self.timeout_per_model//3600}h per model / {self.total_time_budget//3600}h total",
            f"  RG Blocks: {'on' if self.run_research_grade_blocks else 'off'}",
        ]
        return "\n".join(lines)
