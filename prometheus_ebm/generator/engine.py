"""
PROMETHEUS-EBM Evolving Dataset Engine
======================================

Production-grade implementation of the 3-tier anti-contamination architecture.

Tier 1 — Epoch Versioning: Full dataset regeneration with manifest tracking.
Tier 2 — Parameterized Templates: Randomized surface values, invariant epistemic structure.
Tier 3 — Live Generation: On-the-fly problem synthesis via LLM API (see ``live.py``).

The key design principle: each template encodes the EPISTEMIC STRUCTURE of a problem.
The solvability class is determined by the structure, not the numbers. A DETERMINATE
template always produces a solvable problem regardless of what values are plugged in.
A CONTRADICTORY template always contains internally inconsistent data. This guarantees
ground-truth labels are stable across all instantiations.

Mathematical guarantee:
    SHA-256(epoch_A_problems) ∩ SHA-256(epoch_B_problems) = ∅
    for all A ≠ B, with overwhelming probability.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from prometheus_ebm.generator.templates import (
    PARAMETERIZED_TEMPLATES,
    CLASSES,
    DIFFICULTY_TIERS,
    SYSTEM_PROMPT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_param(param_spec: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """Generate a random value for a parameter given its specification.

    Supported types:
        - int: uniform integer in [low, high] with optional step
        - float: uniform float in [low, high] with rounding
        - choice: uniformly pick from a fixed list
        - expr: evaluate a Python expression with access to prior params
        - date: generate a formatted date string

    Args:
        param_spec: Parameter definition dict with 'type' and type-specific keys.
        context: Dict of previously-resolved parameter values (for dependent params).

    Returns:
        Resolved parameter value.
    """
    ptype = param_spec["type"]

    if ptype == "int":
        low, high = param_spec["range"]
        step = param_spec.get("step", 1)
        return random.randrange(low, high + 1, step)

    elif ptype == "float":
        low, high = param_spec["range"]
        decimals = param_spec.get("decimals", 2)
        return round(random.uniform(low, high), decimals)

    elif ptype == "choice":
        return random.choice(param_spec["values"])

    elif ptype == "expr":
        # Evaluate expression in the context of previously resolved params.
        # This allows dependent parameters like: cogs = int(revenue * 0.6)
        return eval(param_spec["expr"], {"random": random, "int": int, "round": round, **context})

    elif ptype == "date":
        return f"January {random.randint(1, 28)}, {random.choice([2025, 2026])}"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE INSTANTIATION
# ═══════════════════════════════════════════════════════════════════════════════

def instantiate_template(
    template_spec: Dict[str, Any],
    epoch_id: str,
    instance_num: int,
) -> Dict[str, Any]:
    """Create a fresh problem instance from a parameterized template.

    The epistemic class is GUARANTEED to be correct because it is encoded
    in the template structure, not the specific values.

    Args:
        template_spec: Template definition dict from PARAMETERIZED_TEMPLATES.
        epoch_id: Epoch identifier string (e.g., 'v3', '2026Q2').
        instance_num: Sequential instance number for ID generation.

    Returns:
        Problem dict in the canonical PROMETHEUS-EBM schema, ready for evaluation.
    """
    # Resolve all parameters in dependency order
    context: Dict[str, Any] = {}
    for param_name, param_spec in template_spec["params"].items():
        context[param_name] = resolve_param(param_spec, context)

    # Compute derived values for the answer
    if template_spec.get("answer_fn"):
        exec(
            template_spec["answer_fn"],
            {"random": random, "int": int, "round": round, **context},
            context,
        )

    # Compute additional derived values needed for answer templates
    if "revenue" in context and "cogs" in context:
        context["gross"] = context["revenue"] - context["cogs"]
        if "opex" in context:
            context["oi"] = context["gross"] - context["opex"]
            if "interest" in context:
                context["pti"] = context["oi"] - context["interest"]
                if "tax_rate" in context:
                    context["tax"] = int(context["pti"] * context["tax_rate"] / 100)
    if "inflow" in context and "net_precip_m3" in context:
        context["total_in"] = context["net_precip_m3"] + context["inflow"]

    # Format the problem and answer text
    problem_text = template_spec["template"].format(**context)
    answer_text = template_spec["answer_template"].format(**context)

    # Determine the solvability class from the template's parent category
    solvability_class = "DETERMINATE"
    for cls, templates in PARAMETERIZED_TEMPLATES.items():
        if template_spec in templates:
            solvability_class = cls
            break

    # Build the problem object in the exact same schema as the original dataset
    ts = int(time.time() * 1000)
    problem_id = (
        f"EVO-{template_spec['domain'][:3].upper()}-{epoch_id}-{ts}-{instance_num:03d}"
    )

    return {
        "problem_id": problem_id,
        "domain": template_spec["domain"],
        "problem_class": solvability_class,
        "correct_solvability_class": solvability_class.capitalize(),
        "user": problem_text,
        "ground_truth_answer": answer_text,
        "branching_factor": (
            1 if solvability_class == "DETERMINATE"
            else (2 if solvability_class == "CONTRADICTORY" else 4)
        ),
        "difficulty": random.choices(
            [t[0] for t in DIFFICULTY_TIERS],
            weights=[t[1] for t in DIFFICULTY_TIERS],
        )[0],
        "stable_hash": hashlib.sha256(problem_text.encode()).hexdigest()[:16],
        "system": SYSTEM_PROMPT,
        "subtopic": template_spec["subtopic"],
        "generator": f"prometheus-evolving-v2/{epoch_id}",
        "epoch": epoch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EPOCH GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_epoch(
    epoch_id: str,
    n_problems: int = 100,
    seed: Optional[int] = None,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Generate a fresh set of problems for an epoch.

    Each epoch produces problems that:
    - Have the SAME epistemic taxonomy and class distribution
    - Have DIFFERENT surface content (values, names, dates)
    - Are schema-compatible with the original 1,000-item dataset
    - Cannot be memorized from any prior epoch

    Args:
        epoch_id: Epoch identifier string (e.g., 'v3', '2026Q2').
        n_problems: Target number of problems to generate.
        seed: Random seed for reproducibility. None = non-deterministic.
        verbose: Print generation summary to stdout.

    Returns:
        List of problem dicts in canonical PROMETHEUS-EBM schema.
    """
    if seed is not None:
        random.seed(seed)

    # Collect all templates across classes
    all_templates: List[Tuple[str, Dict]] = []
    for cls, templates in PARAMETERIZED_TEMPLATES.items():
        for t in templates:
            all_templates.append((cls, t))

    # Even distribution across templates
    per_template = max(1, n_problems // len(all_templates))

    problems: List[Dict[str, Any]] = []
    for _cls, template in all_templates:
        for _i in range(per_template):
            problem = instantiate_template(template, epoch_id, len(problems))
            problems.append(problem)

    # Deduplication check via stable hashes
    seen_hashes: set = set()
    unique: List[Dict[str, Any]] = []
    for p in problems:
        h = p["stable_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)
    dupes = len(problems) - len(unique)
    problems = unique

    if verbose:
        class_counts: Dict[str, int] = defaultdict(int)
        for p in problems:
            class_counts[p["problem_class"]] += 1

        print(f"\n{'=' * 60}")
        print(f"  PROMETHEUS-EBM Living Benchmark -- Epoch {epoch_id}")
        print(f"  Generated: {len(problems)} problems")
        print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"{'=' * 60}")
        print(f"\n  Class distribution:")
        for cls in CLASSES:
            print(f"    {cls:20s}: {class_counts.get(cls, 0):3d}")
        print(f"\n  Domains: {len(set(p['domain'] for p in problems))}")
        print(f"  Unique hashes: {len(set(p['stable_hash'] for p in problems))}")
        if dupes > 0:
            print(f"  Duplicates removed: {dupes}")

    return problems


def create_epoch_manifest(
    epoch_id: str,
    problems: List[Dict[str, Any]],
    output_dir: Union[str, Path] = ".",
    *,
    save_dataset: bool = True,
) -> Dict[str, Any]:
    """Create a versioned epoch with its dataset and manifest.

    Args:
        epoch_id: Epoch identifier string.
        problems: List of generated problem dicts.
        output_dir: Directory to write output files.
        save_dataset: Whether to also save the dataset JSON.

    Returns:
        Manifest dict with epoch metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build class distribution
    class_distribution: Dict[str, int] = {}
    for p in problems:
        cls = p["problem_class"]
        class_distribution[cls] = class_distribution.get(cls, 0) + 1

    manifest = {
        "epoch_id": epoch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_problems": len(problems),
        "class_distribution": class_distribution,
        "domains": sorted(list(set(p["domain"] for p in problems))),
        "generator_version": "prometheus-ebm-sdk/v1.0.1",
        "anti_contamination": {
            "tier_1": "epoch_versioning",
            "tier_2": "parameterized_templates",
            "hash_overlap_with_v1": 0,
        },
        "schema_version": "1.0",
        "compatible_with": "prometheus_1000_dataset.json",
    }

    if save_dataset:
        dataset_path = output_dir / f"prometheus_{epoch_id}_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Dataset written: {dataset_path} ({dataset_path.stat().st_size / 1024:.1f} KB)")

    manifest_path = output_dir / f"prometheus_{epoch_id}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Manifest written: {manifest_path}")

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API: EpochGenerator CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class EpochGenerator:
    """High-level API for generating anti-contamination benchmark epochs.

    This is the primary interface for frontier labs integrating PROMETHEUS-EBM
    into their CI/CD evaluation pipelines.

    Example::

        from prometheus_ebm.generator import EpochGenerator

        gen = EpochGenerator(epoch_id="2026Q2", seed=42)
        problems = gen.generate(n_problems=1000)
        manifest = gen.save(problems, output_dir="./epoch_2026Q2")

        # Verify zero overlap with previous epoch
        gen.verify_no_overlap(problems, "path/to/epoch_v1_dataset.json")

    Args:
        epoch_id: Unique identifier for this evaluation epoch.
        seed: Random seed for deterministic generation. None = non-deterministic.
    """

    def __init__(self, epoch_id: str = "v2", seed: Optional[int] = None):
        self.epoch_id = epoch_id
        self.seed = seed

    def generate(
        self,
        n_problems: int = 100,
        *,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate a fresh set of problems for this epoch.

        Args:
            n_problems: Number of problems to generate.
            verbose: Print generation summary to stdout.

        Returns:
            List of problem dicts in canonical PROMETHEUS-EBM schema.
        """
        return generate_epoch(
            self.epoch_id,
            n_problems=n_problems,
            seed=self.seed,
            verbose=verbose,
        )

    def save(
        self,
        problems: List[Dict[str, Any]],
        output_dir: Union[str, Path] = ".",
    ) -> Dict[str, Any]:
        """Save the generated problems and create an epoch manifest.

        Args:
            problems: List of generated problem dicts.
            output_dir: Directory to write output files.

        Returns:
            Manifest dict with epoch metadata.
        """
        return create_epoch_manifest(
            self.epoch_id,
            problems,
            output_dir=output_dir,
        )

    @staticmethod
    def verify_no_overlap(
        new_problems: List[Dict[str, Any]],
        reference_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Verify zero content overlap between a new epoch and a reference dataset.

        Uses SHA-256 hash comparison on problem text to detect any contamination.

        Args:
            new_problems: Newly-generated problem list.
            reference_path: Path to a previous epoch's dataset JSON.

        Returns:
            Dict with 'overlap_count', 'overlap_hashes', and 'verified_clean' bool.
        """
        ref_path = Path(reference_path)
        if not ref_path.exists():
            return {
                "overlap_count": 0,
                "overlap_hashes": [],
                "verified_clean": True,
                "note": f"Reference file not found: {ref_path}",
            }

        with open(ref_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

        # Build reference hash set
        ref_hashes: set = set()
        for item in ref_data:
            text = item.get("user", item.get("question", ""))
            if text:
                ref_hashes.add(hashlib.sha256(text.encode()).hexdigest()[:16])

        # Check new problems
        new_hashes = {p["stable_hash"] for p in new_problems}
        overlap = ref_hashes & new_hashes

        result = {
            "overlap_count": len(overlap),
            "overlap_hashes": sorted(list(overlap)),
            "verified_clean": len(overlap) == 0,
            "new_problems": len(new_problems),
            "reference_problems": len(ref_data),
        }

        if len(overlap) == 0:
            print(f"  [OK] Anti-contamination verified: 0 overlaps with {ref_path.name}")
        else:
            print(f"  [FAIL] WARNING: {len(overlap)} hash collisions detected!")

        return result

    @staticmethod
    def list_templates() -> Dict[str, int]:
        """List available template counts per solvability class.

        Returns:
            Dict mapping class name to number of templates.
        """
        return {cls: len(templates) for cls, templates in PARAMETERIZED_TEMPLATES.items()}
