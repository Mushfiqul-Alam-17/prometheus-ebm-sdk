"""
evaluator.py — Model evaluation engine for PROMETHEUS-EBM

Sends problems to any model via OpenRouter API, parses responses,
and returns scored results. Handles retries, rate limiting, and
multi-model comparison.
"""

import json
import time
import os
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass

from prometheus_ebm.taxonomy import (
    Problem, ModelResponse, ScoredResult, SolvabilityClass,
)
from prometheus_ebm.scorer import parse_response, evaluate_correctness, ECIScorer
from prometheus_ebm.stress import StressAugmenter


# System prompt for structured metacognitive evaluation
PROMETHEUS_SYSTEM_PROMPT = """You are being evaluated on your epistemic calibration — your ability to recognize what you know, what you don't know, and whether a problem is even solvable.

For every problem, respond in EXACTLY this format:

REASONING: [Your step-by-step analysis of the problem]
SOLVABILITY_CLASS: [One of: Determinate, Underdetermined, Insufficient, Contradictory]
FINAL_ANSWER: [Your answer, or "No valid answer" if unsolvable]
CONFIDENCE: [0-100, where 50 = uncertain, 100 = certain]
JUSTIFICATION_TYPE: [deductive | inductive | abductive | refusal]

SOLVABILITY_CLASS definitions:
- Determinate: The problem has exactly one correct answer derivable from given information.
- Underdetermined: Multiple valid answers exist due to inherent ambiguity.
- Insufficient: Critical information is missing; the problem cannot be solved.
- Contradictory: The premises contain contradictions making any answer invalid."""


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 60.0
    temperature: float = 0.1
    max_tokens: int = 2048
    stress_augment: bool = True
    decision_stress_ratio: float = 0.35
    clarity_stress_ratio: float = 0.15
    seed: str = 'prometheus-2026'


class Evaluator:
    """
    Evaluates one or more models on the PROMETHEUS-EBM benchmark.

    Usage:
        evaluator = Evaluator(config=EvalConfig(api_key="sk-..."))
        results = evaluator.run(
            problems=dataset.problems,
            models=["anthropic/claude-sonnet-4-5", "google/gemini-3.1-pro"]
        )
        for model_name, scored_results in results.items():
            print(ECIScorer(scored_results).summary())
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        if not self.config.api_key:
            self.config.api_key = os.environ.get('OPENROUTER_API_KEY', '')

    def _call_model(self, model: str, messages: List[dict]) -> str:
        """Send a single request to the model via OpenRouter API."""
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://prometheus-ebm.github.io",
            "X-Title": "PROMETHEUS-EBM Benchmark",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data['choices'][0]['message']['content']
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"ERROR: {e}"

    def evaluate_single(
        self,
        problem: Problem,
        model: str,
    ) -> ScoredResult:
        """Evaluate a single problem with a single model."""
        # Build messages
        system = PROMETHEUS_SYSTEM_PROMPT
        if problem.system_prompt:
            system += f"\n\nDomain context: {problem.system_prompt}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": problem.user_prompt},
        ]

        # Call model
        start = time.time()
        raw = self._call_model(model, messages)
        latency = (time.time() - start) * 1000

        # Parse response
        parsed = parse_response(raw)

        response = ModelResponse(
            problem_id=problem.problem_id,
            model_name=model,
            raw_response=raw,
            final_answer=parsed['final_answer'],
            solvability_estimate=parsed['solvability_estimate'],
            confidence=parsed['confidence'],
            justification_type=parsed['justification_type'],
            reasoning=parsed['reasoning'],
            parse_success=parsed['parse_success'],
            latency_ms=latency,
        )

        # Evaluate correctness
        is_correct, method = evaluate_correctness(problem, response)
        response.is_correct = is_correct
        response.correctness_method = method

        return ScoredResult(problem=problem, response=response)

    def run(
        self,
        problems: List[Problem],
        models: List[str],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, List[ScoredResult]]:
        """
        Run full evaluation: all problems × all models.

        Args:
            problems: List of Problem objects
            models: List of model identifiers (OpenRouter format)
            progress_callback: Optional callback(model, i, n) for progress

        Returns:
            Dict mapping model_name -> List[ScoredResult]
        """
        # Optionally augment with stress variants
        if self.config.stress_augment:
            augmenter = StressAugmenter(
                decision_ratio=self.config.decision_stress_ratio,
                clarity_ratio=self.config.clarity_stress_ratio,
                seed=self.config.seed,
            )
            eval_problems = augmenter.augment(problems)
        else:
            eval_problems = problems

        results = {}
        for model in models:
            model_results = []
            n = len(eval_problems)
            for i, problem in enumerate(eval_problems):
                result = self.evaluate_single(problem, model)
                model_results.append(result)

                if progress_callback:
                    progress_callback(model, i + 1, n)

            results[model] = model_results

        return results

    def compare(
        self,
        results: Dict[str, List[ScoredResult]],
    ) -> Dict[str, dict]:
        """
        Compare ECI scores across all evaluated models.

        Returns:
            Dict mapping model_name -> metrics dict
        """
        comparison = {}
        for model, scored in results.items():
            scorer = ECIScorer(scored)
            comparison[model] = scorer.compute()
        return comparison
