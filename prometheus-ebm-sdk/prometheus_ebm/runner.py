"""
PROMETHEUS-EBM Runner — Orchestrates benchmark execution.

Supports two modes:
- COMPARE: Multi-model leaderboard (evaluate N models on the same dataset)
- DEEP_PROBE: Single-model diagnostic (evaluate 1 model on 1,000+ items)
"""
import os
import time
import json
import pandas as pd
from typing import List, Dict, Optional
from prometheus_ebm.config import RunConfig
from prometheus_ebm.scorer import ECIScorer, BrierDecomposition, Type2DPrime


class PrometheusRunner:
    """Main benchmark runner for PROMETHEUS-EBM.
    
    Usage:
        config = RunConfig(mode="compare", models=["claude-opus-4.6", "gemini-3.1"])
        runner = PrometheusRunner(config)
        results = runner.run()
        results.export("output_report.html")
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.config.validate()
        self.scorer = ECIScorer()
        self.results: Optional["BenchmarkResults"] = None
        self._start_time = None
    
    def run(self) -> "BenchmarkResults":
        """Execute the full benchmark pipeline."""
        self._start_time = time.time()
        
        if self.config.verbose:
            print(self.config.summary())
        
        # 1. Load dataset
        dataset = self._load_dataset()
        
        # 2. Apply stress augmentation
        dataset = self._augment_dataset(dataset)
        
        # 3. Resolve model providers
        providers = self._resolve_providers()
        
        # 4. Run evaluation loop
        raw_results = self._evaluate(dataset, providers)
        
        # 5. Score results
        scored = self._score(raw_results)
        
        # 6. Statistical validation
        if self.config.run_statistics:
            stats = self._bootstrap(scored)
        else:
            stats = None
        
        self.results = BenchmarkResults(
            config=self.config,
            model_scores=scored,
            statistics=stats,
            elapsed_seconds=time.time() - self._start_time,
        )
        
        return self.results
    
    def _load_dataset(self) -> List[Dict]:
        """Load the base problem set."""
        path = self.config.dataset_path
        if path is None:
            # Use bundled dataset
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(pkg_dir, "data", "prometheus_200_multimodel_dataset.json")
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Download from: https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk"
            )
        
        with open(path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        # Truncate to n_items if needed
        if self.config.n_items < len(problems):
            problems = problems[:self.config.n_items]
        
        if self.config.verbose:
            print(f"  Loaded {len(problems)} base problems")
        
        return problems
    
    def _augment_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Apply stress augmentation to create challenge variants."""
        # Placeholder — will be implemented with full stress logic
        if self.config.verbose:
            print(f"  Stress augmentation: {len(dataset)} total prompts")
        return dataset
    
    def _resolve_providers(self) -> Dict:
        """Resolve model API providers based on config."""
        provider_name = self.config.provider
        
        if provider_name == "kaggle":
            from prometheus_ebm.providers.kaggle import KaggleProvider
            return KaggleProvider(self.config)
        elif provider_name == "openrouter":
            from prometheus_ebm.providers.openrouter import OpenRouterProvider
            return OpenRouterProvider(self.config)
        elif provider_name == "anthropic":
            from prometheus_ebm.providers.anthropic import AnthropicProvider
            return AnthropicProvider(self.config)
        elif provider_name == "openai":
            from prometheus_ebm.providers.openai import OpenAIProvider
            return OpenAIProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def _evaluate(self, dataset, providers) -> pd.DataFrame:
        """Run evaluation loop across all models."""
        # Placeholder — will be wired to actual API calls
        raise NotImplementedError("Evaluation loop coming in v0.2.0")
    
    def _score(self, raw_results: pd.DataFrame) -> Dict:
        """Score raw results with ECI, Brier, D-Prime."""
        raise NotImplementedError("Scoring pipeline coming in v0.2.0")
    
    def _bootstrap(self, scored: Dict) -> Dict:
        """Bootstrap confidence intervals and significance tests."""
        raise NotImplementedError("Bootstrap analysis coming in v0.2.0")


class BenchmarkResults:
    """Container for benchmark results with export capabilities."""
    
    def __init__(self, config, model_scores, statistics, elapsed_seconds):
        self.config = config
        self.model_scores = model_scores
        self.statistics = statistics
        self.elapsed_seconds = elapsed_seconds
    
    def export(self, path: str, format: str = "auto"):
        """Export results to file.
        
        Args:
            path: Output file path
            format: "csv", "json", "html", or "auto" (inferred from extension)
        """
        if format == "auto":
            ext = os.path.splitext(path)[1].lower()
            format = ext.lstrip(".")
        
        if format == "csv":
            self._export_csv(path)
        elif format == "json":
            self._export_json(path)
        elif format == "html":
            self._export_html(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, path):
        """Export as CSV."""
        # Placeholder
        pass
    
    def _export_json(self, path):
        """Export as JSON."""
        # Placeholder
        pass
    
    def _export_html(self, path):
        """Export as HTML report."""
        # Placeholder
        pass
