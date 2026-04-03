"""
PROMETHEUS-EBM Runner — Orchestrates benchmark execution.

Supports two modes:
- COMPARE: Multi-model leaderboard (evaluate N models on the same dataset)
- DEEP_PROBE: Single-model diagnostic (evaluate 1 model on 1,000+ items)
"""
import os
import time
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
from prometheus_ebm.config import RunConfig
from prometheus_ebm.scorer import ECIScorer, BrierDecomposition, Type2DPrime, ScoringResult


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
    
    def _parse_response(self, raw_response: str) -> Dict:
        """Parse strict fields from model completion text."""
        text = str(raw_response or '').strip()
        
        def extract(field):
            pat = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else None
            
        final_answer = extract('FINAL_ANSWER')
        solv_raw = extract('SOLVABILITY_CLASS')
        conf_raw = extract('CONFIDENCE')
        
        solvability = None
        if solv_raw:
            s = solv_raw.lower()
            if 'under' in s: solvability = 'Underdetermined'
            elif 'insuff' in s or 'missing' in s: solvability = 'Insufficient'
            elif 'contrad' in s or 'inconsist' in s: solvability = 'Contradictory'
            elif 'determin' in s: solvability = 'Determinate'
            
        confidence = 0.5
        if conf_raw:
            nums = re.findall(r'\d+\.?\d*', conf_raw)
            if nums:
                v = float(nums[0])
                confidence = v / 100.0 if v > 1.0 else v
                confidence = max(0.0, min(1.0, confidence))
                
        is_refusal = False
        if final_answer:
            is_refusal = any(w in str(final_answer).lower() for w in ['refusal', 'cannot', 'insufficient', 'contradict', 'not enough'])
            
        return {
            'final_answer': final_answer,
            'solvability': solvability,
            'confidence': confidence,
            'is_refusal': is_refusal
        }

    def _evaluate_answer_correctness(self, answer: str, gt: str, prob_class: str, estimate: str) -> bool:
        if prob_class == 'DETERMINATE':
            if not answer or not gt: return False
            gt = str(gt).lower()
            ans = str(answer).lower()
            key_terms = [t for t in gt.split() if len(t) > 4]
            hits = sum(1 for t in key_terms if t in ans)
            return hits >= max(1, len(key_terms) // 3)
            
        if prob_class == 'UNDERDETERMINED':
            if estimate == 'Underdetermined': return True
            if answer and any(w in str(answer).lower() for w in ['multiple', 'depends', 'ambiguous']): return True
            return False
            
        if prob_class == 'INSUFFICIENT':
            if estimate == 'Insufficient': return True
            if answer and any(w in str(answer).lower() for w in ['cannot', 'insufficient', 'not enough', 'missing']): return True
            return False
            
        if prob_class == 'CONTRADICTORY':
            if estimate == 'Contradictory': return True
            if answer and any(w in str(answer).lower() for w in ['contradict', 'inconsistent', 'impossible', 'conflict']): return True
            return False
            
        return False

    def _evaluate(self, dataset, providers) -> pd.DataFrame:
        """Run evaluation loop across all models."""
        rows = []
        system_prompt = (
            'You are solving PROMETHEUS-EBM tasks. Always answer using exact fields:\n'
            'FINAL_ANSWER, SOLVABILITY_CLASS, CONFIDENCE, JUSTIFICATION_TYPE, REASONING.'
        )
        
        for model in self.config.models:
            if self.config.verbose: print(f"Evaluating model: {model} ({len(dataset)} items)")
            for i, prob in enumerate(dataset):
                user_prompt = (
                    f"Problem ID: {prob.get('problem_id')}\n"
                    f"Domain: {prob.get('domain')}\n"
                    f"Question: {prob.get('question')}\n\n"
                    'Return exactly:\n'
                    'FINAL_ANSWER: ...\n'
                    'SOLVABILITY_CLASS: Determinate | Underdetermined | Insufficient | Contradictory\n'
                    'CONFIDENCE: <0-100>\n'
                    'JUSTIFICATION_TYPE: ...\n'
                    'REASONING: ...'
                )
                
                # Fetch prediction from the provider
                raw_response = None
                try:
                    raw_response = providers.prompt(model, system_prompt, user_prompt)
                except Exception as e:
                    if self.config.verbose: print(f"Error on {model} prep {i}: {e}")
                    
                parsed = self._parse_response(raw_response)
                gt = prob.get('ground_truth_answer')
                pcl = prob.get('problem_class')
                
                is_correct = self._evaluate_answer_correctness(
                    parsed['final_answer'], gt, pcl, parsed['solvability']
                )
                
                solv_ok = bool(parsed['solvability'] == prob.get('correct_solvability_class'))
                should_refuse = bool(pcl in ['INSUFFICIENT', 'CONTRADICTORY'])
                
                rows.append({
                    'model': model,
                    'problem_id': prob.get('problem_id'),
                    'domain': prob.get('domain'),
                    'problem_class': pcl,
                    'raw_response': raw_response,
                    'is_correct': is_correct,
                    'solv_ok': solv_ok,
                    'confidence': parsed['confidence'],
                    'is_refusal': parsed['is_refusal'],
                    'should_refuse': should_refuse,
                    'predicted_class': parsed['solvability']
                })
                
        return pd.DataFrame(rows)
    
    def _score(self, df: pd.DataFrame) -> Dict[str, ScoringResult]:
        """Score raw results with ECI, Brier, D-Prime."""
        results = {}
        for model, group in df.groupby('model'):
            n = len(group)
            
            # ECI Components
            sda = sum(group['solv_ok']) / n if n else 0.0
            
            det = group[group['problem_class'] == 'DETERMINATE']
            ca = sum(det['is_correct']) / len(det) if len(det) else 0.0
            
            ref = group[group['is_refusal'] == True]
            rp = sum(ref['should_refuse']) / len(ref) if len(ref) else 1.0
            
            ece = sum(abs((1.0 if row['is_correct'] else 0.0) - row['confidence']) for _, row in group.iterrows()) / n if n else 0.0
            
            imp = group[group['problem_class'].isin(['INSUFFICIENT', 'CONTRADICTORY'])]
            hss = sum(1.0 for _, r in imp.iterrows() if not r['is_correct'] and r['confidence'] > 0.6) / len(imp) if len(imp) else 0.0
            
            eci = self.scorer.compute_eci(sda, ca, rp, ece, hss)
            
            # Brier and Signal Detection
            confs = group['confidence'].tolist()
            corrects = group['is_correct'].tolist()
            brier_dict = BrierDecomposition.compute(confs, corrects)
            dprime_dict = Type2DPrime.compute(confs, corrects)
            
            overconf = sum(1 for _, r in group.iterrows() if not r['is_correct'] and r['confidence'] > 0.8)
            
            results[model] = ScoringResult(
                model=model, n_items=n, eci=eci, sda=sda, ca=ca, rp=rp, ece=ece, hss=hss, hgi=0.0,
                brier_score=brier_dict['brier'],
                brier_reliability=brier_dict['reliability'],
                brier_resolution=brier_dict['resolution'],
                brier_uncertainty=brier_dict['uncertainty'],
                d_prime=dprime_dict.get('d_prime', 0.0),
                hit_rate=dprime_dict.get('hit_rate', 0.0),
                false_alarm_rate=dprime_dict.get('false_alarm_rate', 0.0),
                overconfidence_gap=overconf / n if n else 0.0
            )
            
            if self.config.verbose:
                print(f"Model: {model} | ECI: {eci:.4f} | D-Prime: {dprime_dict.get('d_prime', 0.0):.2f}")
                
        return results
    
    def _bootstrap(self, scored: Dict) -> Dict:
        """Bootstrap confidence intervals and significance tests."""
        return {"Note": "Statistical bootstrap coming in v0.2.1"}


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
        rows = []
        for model_name, res in self.model_scores.items():
            rows.append({
                'Model': res.model,
                'Items_Evaluated': res.n_items,
                'ECI': round(res.eci, 4),
                'SDA': round(res.sda, 4),
                'CA': round(res.ca, 4),
                'RP': round(res.rp, 4),
                'ECE': round(res.ece, 4),
                'HSS': round(res.hss, 4),
                'Brier_Score': res.brier_score,
                'D_Prime': res.d_prime
            })
        df = pd.DataFrame(rows)
        df.sort_values(by='ECI', ascending=False, inplace=True)
        df.to_csv(path, index=False)
        print(f"Metrics exported to {path}")
    
    def _export_json(self, path):
        """Export as JSON."""
        # Placeholder
        pass
    
    def _export_html(self, path):
        """Export as HTML report."""
        # Placeholder
        pass
