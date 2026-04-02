"""
dataset.py — Dataset loading and management for PROMETHEUS-EBM

Loads the built-in 200-problem real-world dataset or custom datasets.
Handles domain/class filtering, random sampling, and reproducible splits.
"""

import json
import os
from typing import List, Optional, Dict
from pathlib import Path

from prometheus_ebm.taxonomy import (
    SolvabilityClass, Domain, RigorMode, Problem,
)


# Path to built-in dataset
_DATA_DIR = Path(__file__).parent / 'data'
_DEFAULT_DATASET = _DATA_DIR / 'prometheus_real_200.json'


class Dataset:
    """
    A collection of PROMETHEUS-EBM evaluation problems.

    Usage:
        # Load built-in dataset
        ds = Dataset.load_default()

        # Load custom dataset
        ds = Dataset.from_json("my_problems.json")

        # Filter
        financial = ds.filter(domain=Domain.FINANCIAL)
        impossible = ds.filter(solvability=[SolvabilityClass.INSUFFICIENT,
                                             SolvabilityClass.CONTRADICTORY])
    """

    def __init__(self, problems: List[Problem]):
        self.problems = problems

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]

    @classmethod
    def load_default(cls) -> 'Dataset':
        """Load the built-in 200-problem real-world dataset."""
        if not _DEFAULT_DATASET.exists():
            raise FileNotFoundError(
                f"Built-in dataset not found at {_DEFAULT_DATASET}. "
                f"Run `prometheus-ebm install-data` or copy prometheus_200_multimodel_dataset.json "
                f"to {_DATA_DIR}/"
            )
        return cls.from_json(str(_DEFAULT_DATASET))

    @classmethod
    def from_json(cls, path: str) -> 'Dataset':
        """
        Load a dataset from a JSON file.

        Expected format: list of objects with fields:
            problem_id, domain, problem_class, system, user/question,
            ground_truth_answer, correct_solvability_class, branching_factor
        """
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        problems = []
        for item in raw:
            try:
                # Parse domain
                domain_str = item.get('domain', 'medical')
                try:
                    domain = Domain.from_string(domain_str)
                except ValueError:
                    domain = Domain.MEDICAL

                # Parse solvability class
                cls_str = item.get('correct_solvability_class',
                                   item.get('problem_class', 'DETERMINATE'))
                try:
                    solv_class = SolvabilityClass.from_string(cls_str)
                except ValueError:
                    solv_class = SolvabilityClass.DETERMINATE

                problem = Problem(
                    problem_id=item['problem_id'],
                    domain=domain,
                    solvability_class=solv_class,
                    system_prompt=item.get('system', ''),
                    user_prompt=item.get('user', item.get('question', '')),
                    ground_truth_answer=item.get('ground_truth_answer'),
                    branching_factor=item.get('branching_factor', 2),
                    rigor_mode=RigorMode.BASE,
                    metadata=item.get('metadata', {}),
                )
                problems.append(problem)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping problem {item.get('problem_id', '?')}: {e}")

        return cls(problems)

    def filter(
        self,
        domain: Optional[Domain] = None,
        solvability: Optional[List[SolvabilityClass]] = None,
        rigor_mode: Optional[RigorMode] = None,
    ) -> 'Dataset':
        """Filter problems by domain, solvability class, or rigor mode."""
        filtered = self.problems

        if domain is not None:
            filtered = [p for p in filtered if p.domain == domain]
        if solvability is not None:
            filtered = [p for p in filtered if p.solvability_class in solvability]
        if rigor_mode is not None:
            filtered = [p for p in filtered if p.rigor_mode == rigor_mode]

        return Dataset(filtered)

    def by_class(self) -> Dict[SolvabilityClass, 'Dataset']:
        """Split into sub-datasets by solvability class."""
        out = {}
        for cls in SolvabilityClass:
            items = [p for p in self.problems if p.solvability_class == cls]
            if items:
                out[cls] = Dataset(items)
        return out

    def by_domain(self) -> Dict[Domain, 'Dataset']:
        """Split into sub-datasets by domain."""
        out = {}
        for dom in Domain:
            items = [p for p in self.problems if p.domain == dom]
            if items:
                out[dom] = Dataset(items)
        return out

    def stats(self) -> dict:
        """Summary statistics about the dataset."""
        from collections import Counter
        class_counts = Counter(p.solvability_class.name for p in self.problems)
        domain_counts = Counter(p.domain.value for p in self.problems)
        return {
            'total': len(self.problems),
            'by_class': dict(class_counts),
            'by_domain': dict(domain_counts),
        }

    def to_json(self, path: str):
        """Save dataset to JSON file."""
        data = [p.to_dict() for p in self.problems]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
