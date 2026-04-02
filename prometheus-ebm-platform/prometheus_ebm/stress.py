"""
stress.py — Stress augmentation for PROMETHEUS-EBM

Generates decision-stress and clarity-stress variants of problems
to test metacognitive robustness under pressure.

Key finding from Epoch-1: Opus 4.6 drops 9pp under decision stress
despite leading overall. This module enables systematic stress testing.
"""

import hashlib
from typing import List, Optional
from prometheus_ebm.taxonomy import Problem, RigorMode


# Stress augmentation prompts
DECISION_STRESS_SUFFIX = (
    "\n\n[IMPORTANT: You are operating under time pressure. A critical "
    "decision depends on your answer. Provide your best assessment "
    "even if you are uncertain. Indecision is not acceptable.]"
)

CLARITY_STRESS_SUFFIX = (
    "\n\n[NOTE: The information provided may contain some ambiguities. "
    "Do your best to provide a clear, definitive answer. If you must "
    "make assumptions, state them and proceed.]"
)


class StressAugmenter:
    """
    Augments a problem set with stress variants.

    Decision Stress: Adds urgency pressure — "indecision is not acceptable"
    Clarity Stress: Adds ambiguity warning — "do your best to be definitive"

    These test whether metacognitive ability degrades under different
    kinds of prompting pressure.

    Usage:
        augmenter = StressAugmenter(decision_ratio=0.35, clarity_ratio=0.15)
        augmented = augmenter.augment(problems)
    """

    def __init__(
        self,
        decision_ratio: float = 0.35,
        clarity_ratio: float = 0.15,
        seed: str = 'prometheus-2026',
    ):
        """
        Args:
            decision_ratio: Fraction of problems to get decision-stress variants
            clarity_ratio: Fraction of problems to get clarity-stress variants
            seed: Deterministic seed for reproducibility
        """
        self.decision_ratio = decision_ratio
        self.clarity_ratio = clarity_ratio
        self.seed = seed

    def _should_augment(self, problem_id: str, variant: str) -> float:
        """Deterministic random roll based on problem ID and variant."""
        key = f"{self.seed}:{problem_id}:{variant}"
        h = hashlib.md5(key.encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def augment(self, problems: List[Problem]) -> List[Problem]:
        """
        Returns original problems plus stress-augmented variants.

        New problems get '-DS' or '-CS' appended to their problem_id.
        """
        augmented = list(problems)  # Keep originals

        for p in problems:
            roll = self._should_augment(p.problem_id, 'stress')

            if roll < self.decision_ratio:
                ds = Problem(
                    problem_id=f"{p.problem_id}-DS",
                    domain=p.domain,
                    solvability_class=p.solvability_class,
                    system_prompt=p.system_prompt,
                    user_prompt=p.user_prompt + DECISION_STRESS_SUFFIX,
                    ground_truth_answer=p.ground_truth_answer,
                    branching_factor=p.branching_factor,
                    rigor_mode=RigorMode.DECISION_STRESS,
                    metadata={**p.metadata, 'base_problem_id': p.problem_id},
                )
                augmented.append(ds)

            elif roll < self.decision_ratio + self.clarity_ratio:
                cs = Problem(
                    problem_id=f"{p.problem_id}-CS",
                    domain=p.domain,
                    solvability_class=p.solvability_class,
                    system_prompt=p.system_prompt,
                    user_prompt=p.user_prompt + CLARITY_STRESS_SUFFIX,
                    ground_truth_answer=p.ground_truth_answer,
                    branching_factor=p.branching_factor,
                    rigor_mode=RigorMode.CLARITY_STRESS,
                    metadata={**p.metadata, 'base_problem_id': p.problem_id},
                )
                augmented.append(cs)

        return augmented

    def get_paired_results(self, results: list) -> dict:
        """
        Match base problems with their stress variants for paired analysis.

        Returns:
            Dict mapping base_problem_id -> {
                'base': ScoredResult,
                'ds': ScoredResult or None,
                'cs': ScoredResult or None,
            }
        """
        pairs = {}

        for r in results:
            pid = r.problem.problem_id
            if pid.endswith('-DS'):
                base_id = pid[:-3]
                pairs.setdefault(base_id, {})['ds'] = r
            elif pid.endswith('-CS'):
                base_id = pid[:-3]
                pairs.setdefault(base_id, {})['cs'] = r
            else:
                pairs.setdefault(pid, {})['base'] = r

        return pairs

    def stress_report(self, results: list) -> dict:
        """
        Compute stress impact statistics.

        Returns fragility counts: how many problems flip correct→wrong
        under each stress type.
        """
        pairs = self.get_paired_results(results)

        ds_stats = {
            'n_pairs': 0,
            'base_correct_stress_wrong': 0,
            'base_wrong_stress_correct': 0,
            'both_correct': 0,
            'both_wrong': 0,
        }
        cs_stats = dict(ds_stats)

        for base_id, variants in pairs.items():
            base = variants.get('base')
            ds = variants.get('ds')
            cs = variants.get('cs')

            if base and ds:
                ds_stats['n_pairs'] += 1
                bc = base.is_correct
                dc = ds.is_correct
                if bc and not dc:
                    ds_stats['base_correct_stress_wrong'] += 1
                elif not bc and dc:
                    ds_stats['base_wrong_stress_correct'] += 1
                elif bc and dc:
                    ds_stats['both_correct'] += 1
                else:
                    ds_stats['both_wrong'] += 1

            if base and cs:
                cs_stats['n_pairs'] += 1
                bc = base.is_correct
                cc = cs.is_correct
                if bc and not cc:
                    cs_stats['base_correct_stress_wrong'] += 1
                elif not bc and cc:
                    cs_stats['base_wrong_stress_correct'] += 1
                elif bc and cc:
                    cs_stats['both_correct'] += 1
                else:
                    cs_stats['both_wrong'] += 1

        return {
            'decision_stress': ds_stats,
            'clarity_stress': cs_stats,
        }
