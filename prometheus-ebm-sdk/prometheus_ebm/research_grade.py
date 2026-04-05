"""Research-grade artifact generation for PROMETHEUS-EBM.

This module mirrors the V5 notebook artifact contract so external labs can run the
SDK directly and produce comparable outputs without notebook-specific glue code.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import statistics
import zipfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int_seed(seed_text: str, bits: int = 31) -> int:
    import hashlib

    digest = hashlib.sha256(str(seed_text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**bits)


def rg_bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 11,
) -> Tuple[float, float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")

    n_boot = max(200, int(n_boot))
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = (1.0 - ci) / 2.0
    lo = means[int(alpha * len(means))]
    hi = means[int((1.0 - alpha) * len(means)) - 1]
    return float(sum(vals) / n), float(lo), float(hi)


def rg_permutation_pvalue(
    a_vals: Sequence[float],
    b_vals: Sequence[float],
    rounds: int = 1000,
    seed: int = 17,
) -> float:
    a = [float(v) for v in a_vals if pd.notna(v)]
    b = [float(v) for v in b_vals if pd.notna(v)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = list(a + b)
    n_a = len(a)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(int(rounds)):
        rng.shuffle(pooled)
        a_star = pooled[:n_a]
        b_star = pooled[n_a:]
        delta = abs(statistics.mean(a_star) - statistics.mean(b_star))
        if delta >= observed:
            extreme += 1

    return (extreme + 1.0) / (rounds + 1.0)


def rg_cohens_h(p1: float, p2: float) -> float:
    p1 = min(max(float(p1), 1e-9), 1.0 - 1e-9)
    p2 = min(max(float(p2), 1e-9), 1.0 - 1e-9)
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def rg_effect_label(h: float) -> str:
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    if ah < 0.5:
        return "small"
    if ah < 0.8:
        return "medium"
    return "large"


def _norm_class(value: object) -> str:
    s = str(value or "").strip().lower()
    if "under" in s:
        return "UNDERDETERMINED"
    if "insuff" in s or "missing" in s:
        return "INSUFFICIENT"
    if "contrad" in s or "inconsist" in s or "conflict" in s:
        return "CONTRADICTORY"
    if "determin" in s:
        return "DETERMINATE"
    return ""


def _norm_bool(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f", ""}:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def _norm_conf(value: object) -> float:
    try:
        c = float(value)
    except Exception:
        c = 0.5
    if c > 1.0:
        c = c / 100.0
    return float(max(0.0, min(1.0, c)))


def _norm_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _ece(confidences: Sequence[float], correctness: Sequence[bool], n_bins: int = 10) -> float:
    if len(confidences) == 0:
        return 0.0
    n = len(confidences)
    ece_val = 0.0
    bin_size = 1.0 / float(n_bins)
    for i in range(int(n_bins)):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        idx = [
            j
            for j, c in enumerate(confidences)
            if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)
        ]
        if not idx:
            continue
        avg_conf = sum(confidences[j] for j in idx) / len(idx)
        avg_acc = sum(1.0 for j in idx if correctness[j]) / len(idx)
        ece_val += len(idx) * abs(avg_conf - avg_acc)
    return float(ece_val / float(n))


def _ensure_epoch1_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "model" not in df.columns:
        df["model"] = "unknown"
    if "problem_id" not in df.columns:
        df["problem_id"] = ""
    if "problem_class" not in df.columns:
        df["problem_class"] = ""
    if "domain" not in df.columns:
        df["domain"] = ""

    if "predicted_class" not in df.columns:
        if "solvability_class" in df.columns:
            df["predicted_class"] = df["solvability_class"]
        else:
            df["predicted_class"] = ""

    if "is_correct" not in df.columns:
        if "correctness_flag" in df.columns:
            df["is_correct"] = df["correctness_flag"]
        else:
            df["is_correct"] = 0

    if "solv_ok" not in df.columns:
        df["solv_ok"] = (
            df["predicted_class"].map(_norm_class)
            == df["problem_class"].map(_norm_class)
        )

    if "confidence" not in df.columns:
        df["confidence"] = 0.5

    if "is_refusal" not in df.columns:
        df["is_refusal"] = df["predicted_class"].map(_norm_class) != "DETERMINATE"

    if "should_refuse" not in df.columns:
        df["should_refuse"] = df["problem_class"].map(_norm_class).isin(
            ["INSUFFICIENT", "CONTRADICTORY"]
        )

    df["problem_class"] = df["problem_class"].map(_norm_class)
    df["predicted_class"] = df["predicted_class"].map(_norm_class)
    df["is_correct"] = df["is_correct"].map(_norm_bool).astype(int)
    df["solv_ok"] = df["solv_ok"].map(_norm_bool).astype(int)
    df["is_refusal"] = df["is_refusal"].map(_norm_bool).astype(int)
    df["should_refuse"] = df["should_refuse"].map(_norm_bool).astype(int)
    df["confidence"] = df["confidence"].map(_norm_conf).astype(float)
    return df


def _compute_epoch1_metrics(df_model: pd.DataFrame) -> Dict[str, float]:
    g = _ensure_epoch1_schema(df_model)
    n = int(len(g))
    if n == 0:
        return {
            "n": 0,
            "eci": float("nan"),
            "sda": float("nan"),
            "ca": float("nan"),
            "rp": float("nan"),
            "ece": float("nan"),
            "hss": float("nan"),
            "hgi": float("nan"),
        }

    sda = float(g["solv_ok"].mean())

    det = g[g["problem_class"] == "DETERMINATE"]
    ca = float(det["is_correct"].mean()) if len(det) else 0.0

    refusals = g[g["predicted_class"] != "DETERMINATE"]
    rp = (
        float((refusals["problem_class"] != "DETERMINATE").mean())
        if len(refusals)
        else 0.0
    )

    confs = g["confidence"].tolist()
    corrects = [bool(v) for v in g["is_correct"].tolist()]
    ece = _ece(confs, corrects)

    impossible = g[g["problem_class"].isin(["INSUFFICIENT", "CONTRADICTORY"])]
    hss = float(1.0 - impossible["is_correct"].mean()) if len(impossible) else 0.0

    eci = 0.30 * sda + 0.25 * ca + 0.20 * rp + 0.15 * (1.0 - ece) + 0.10 * (1.0 - hss)

    expected_conf = (g["is_correct"] + g["solv_ok"]) / 2.0
    hgi = float((g["confidence"] - expected_conf).abs().mean())

    return {
        "n": n,
        "eci": float(eci),
        "sda": float(sda),
        "ca": float(ca),
        "rp": float(rp),
        "ece": float(ece),
        "hss": float(hss),
        "hgi": float(hgi),
    }


def _save_csv(df: pd.DataFrame, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def _save_json(payload: Dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_epoch1_artifacts(
    raw_df: pd.DataFrame,
    seeds: Sequence[str],
    bootstrap_iterations: int,
    pairwise_rounds: int,
    min_seeds_required: int,
    output_dir: str,
) -> Dict[str, bool]:
    source = _ensure_epoch1_schema(raw_df)
    seeds_local = list(seeds) if len(seeds) else ["prometheus-2026-s1"]
    if len(source) == 0:
        for name in [
            "rg_epoch1_seed_summary.csv",
            "rg_epoch1_seed_item_level.csv",
            "rg_epoch1_seed_class_summary.csv",
            "rg_epoch1_seed_domain_summary.csv",
            "rg_epoch1_component_ci.csv",
            "rg_epoch1_eci_hgi_ci.csv",
            "rg_epoch1_pairwise_significance.csv",
            "rg_epoch1_pairwise_significance_hgi.csv",
        ]:
            _save_csv(pd.DataFrame(), output_dir, name)
        return {"RG_EPOCH1_MULTI_SEED_PASS": False}

    epoch1_seed_summaries: List[pd.DataFrame] = []
    epoch1_seed_items: List[pd.DataFrame] = []

    for seed_value in seeds_local:
        sampled_parts: List[pd.DataFrame] = []
        for model_name, g in source.groupby("model", observed=False):
            sampled_parts.append(
                g.sample(
                    n=len(g),
                    replace=True,
                    random_state=stable_int_seed(f"{seed_value}-{model_name}"),
                )
            )
        sampled = (
            pd.concat(sampled_parts, ignore_index=True)
            if sampled_parts
            else source.iloc[0:0].copy()
        )
        sampled["seed"] = seed_value

        summary_rows = []
        for model_name, g in sampled.groupby("model", observed=False):
            metrics = _compute_epoch1_metrics(g)
            summary_rows.append(
                {
                    "seed": seed_value,
                    "model": model_name,
                    "n": int(metrics["n"]),
                    "eci": metrics["eci"],
                    "sda": metrics["sda"],
                    "ca": metrics["ca"],
                    "rp": metrics["rp"],
                    "ece": metrics["ece"],
                    "hss": metrics["hss"],
                    "hgi": metrics["hgi"],
                }
            )

        epoch1_seed_items.append(sampled)
        epoch1_seed_summaries.append(pd.DataFrame(summary_rows))

    rg_epoch1_seed_summary = (
        pd.concat(epoch1_seed_summaries, ignore_index=True)
        if epoch1_seed_summaries
        else pd.DataFrame()
    )
    rg_epoch1_seed_items = (
        pd.concat(epoch1_seed_items, ignore_index=True)
        if epoch1_seed_items
        else pd.DataFrame()
    )

    _save_csv(rg_epoch1_seed_summary, output_dir, "rg_epoch1_seed_summary.csv")
    _save_csv(rg_epoch1_seed_items, output_dir, "rg_epoch1_seed_item_level.csv")

    if len(rg_epoch1_seed_items):
        cls_detail = (
            rg_epoch1_seed_items.groupby(["seed", "model", "problem_class"], observed=False)
            .agg(n=("problem_id", "count"), accuracy=("is_correct", "mean"))
            .reset_index()
        )
        dom_detail = (
            rg_epoch1_seed_items.groupby(["seed", "model", "domain"], observed=False)
            .agg(n=("problem_id", "count"), accuracy=("is_correct", "mean"))
            .reset_index()
        )
    else:
        cls_detail = pd.DataFrame()
        dom_detail = pd.DataFrame()

    _save_csv(cls_detail, output_dir, "rg_epoch1_seed_class_summary.csv")
    _save_csv(dom_detail, output_dir, "rg_epoch1_seed_domain_summary.csv")

    component_metrics = ["eci", "sda", "ca", "rp", "ece", "hss", "hgi"]
    component_rows = []
    for model_name, g in rg_epoch1_seed_summary.groupby("model", observed=False):
        row: Dict[str, object] = {"model": model_name, "n_seeds": int(len(g))}
        for metric in component_metrics:
            mean_v, lo_v, hi_v = rg_bootstrap_mean_ci(
                g[metric].tolist(),
                n_boot=bootstrap_iterations,
                seed=stable_int_seed(f"rg02-{metric}-{model_name}"),
            )
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_ci_low"] = lo_v
            row[f"{metric}_ci_high"] = hi_v
        component_rows.append(row)

    rg_epoch1_component_ci = pd.DataFrame(component_rows)
    _save_csv(rg_epoch1_component_ci, output_dir, "rg_epoch1_component_ci.csv")

    if len(rg_epoch1_component_ci):
        rg_epoch1_ci = rg_epoch1_component_ci[
            [
                "model",
                "eci_mean",
                "eci_ci_low",
                "eci_ci_high",
                "hgi_mean",
                "hgi_ci_low",
                "hgi_ci_high",
                "n_seeds",
            ]
        ].sort_values("eci_mean", ascending=False)
    else:
        rg_epoch1_ci = pd.DataFrame()

    _save_csv(rg_epoch1_ci, output_dir, "rg_epoch1_eci_hgi_ci.csv")

    pair_rows = []
    pair_rows_hgi = []
    unique_models = (
        sorted(rg_epoch1_seed_summary["model"].unique().tolist())
        if len(rg_epoch1_seed_summary)
        else []
    )

    for a, b in itertools.combinations(unique_models, 2):
        a_vals = rg_epoch1_seed_summary[rg_epoch1_seed_summary["model"] == a]["eci"].tolist()
        b_vals = rg_epoch1_seed_summary[rg_epoch1_seed_summary["model"] == b]["eci"].tolist()

        p_val = rg_permutation_pvalue(
            a_vals,
            b_vals,
            rounds=pairwise_rounds,
            seed=stable_int_seed(f"{a}|{b}|eci"),
        )
        h_val = rg_cohens_h(statistics.mean(a_vals), statistics.mean(b_vals))

        pair_rows.append(
            {
                "metric": "eci",
                "model_a": a,
                "model_b": b,
                "mean_a": statistics.mean(a_vals),
                "mean_b": statistics.mean(b_vals),
                "delta_a_minus_b": statistics.mean(a_vals) - statistics.mean(b_vals),
                "permutation_pvalue": p_val,
                "cohens_h": h_val,
                "effect_size": rg_effect_label(h_val),
            }
        )

        a_hgi = rg_epoch1_seed_summary[rg_epoch1_seed_summary["model"] == a]["hgi"].tolist()
        b_hgi = rg_epoch1_seed_summary[rg_epoch1_seed_summary["model"] == b]["hgi"].tolist()
        p_hgi = rg_permutation_pvalue(
            a_hgi,
            b_hgi,
            rounds=pairwise_rounds,
            seed=stable_int_seed(f"{a}|{b}|hgi"),
        )
        pair_rows_hgi.append(
            {
                "metric": "hgi",
                "model_a": a,
                "model_b": b,
                "mean_a": statistics.mean(a_hgi),
                "mean_b": statistics.mean(b_hgi),
                "delta_a_minus_b": statistics.mean(a_hgi) - statistics.mean(b_hgi),
                "permutation_pvalue": p_hgi,
            }
        )

    rg_epoch1_pairwise = (
        pd.DataFrame(pair_rows).sort_values("permutation_pvalue")
        if pair_rows
        else pd.DataFrame()
    )
    rg_epoch1_pairwise_hgi = (
        pd.DataFrame(pair_rows_hgi).sort_values("permutation_pvalue")
        if pair_rows_hgi
        else pd.DataFrame()
    )

    _save_csv(rg_epoch1_pairwise, output_dir, "rg_epoch1_pairwise_significance.csv")
    _save_csv(rg_epoch1_pairwise_hgi, output_dir, "rg_epoch1_pairwise_significance_hgi.csv")

    seed_unique = (
        int(rg_epoch1_seed_summary["seed"].nunique())
        if "seed" in rg_epoch1_seed_summary.columns and len(rg_epoch1_seed_summary)
        else 0
    )
    multi_seed_pass = bool(seed_unique >= min_seeds_required and len(rg_epoch1_seed_summary) > 0)
    return {"RG_EPOCH1_MULTI_SEED_PASS": multi_seed_pass}


def _ensure_probe_schema(df_probe: pd.DataFrame) -> pd.DataFrame:
    d = df_probe.copy()
    d.columns = [str(c).replace("\ufeff", "").strip() for c in d.columns]

    if "correctness_flag" not in d.columns:
        if "is_correct" in d.columns:
            d["correctness_flag"] = d["is_correct"]
        else:
            d["correctness_flag"] = 0
    d["correctness_flag"] = d["correctness_flag"].map(_norm_bool).astype(int)

    if "solvability_class" not in d.columns:
        if "solvability_estimate" in d.columns:
            d["solvability_class"] = d["solvability_estimate"]
        elif "predicted_class" in d.columns:
            d["solvability_class"] = d["predicted_class"]
        else:
            d["solvability_class"] = ""

    d["solvability_class"] = d["solvability_class"].map(
        lambda v: {
            "UNDERDETERMINED": "Underdetermined",
            "INSUFFICIENT": "Insufficient",
            "CONTRADICTORY": "Contradictory",
            "DETERMINATE": "Determinate",
        }.get(_norm_class(v), "")
    )

    if "parse_success" in d.columns:
        d["parse_success"] = d["parse_success"].fillna(False).astype(bool)
    else:
        d["parse_success"] = d["solvability_class"].astype(str).str.strip().ne("")

    d["solvability_present"] = d["solvability_class"].astype(str).str.strip().ne("")

    if "confidence" not in d.columns:
        d["confidence"] = 0.5
    d["confidence"] = d["confidence"].map(_norm_conf).astype(float)

    if "model" not in d.columns:
        d["model"] = "unknown"
    if "problem_class" not in d.columns:
        d["problem_class"] = ""
    if "domain" not in d.columns:
        d["domain"] = ""
    if "problem_id" not in d.columns:
        d["problem_id"] = ""

    d["problem_class"] = d["problem_class"].map(_norm_class)
    return d


def write_epoch2_artifacts(
    probe_df: pd.DataFrame,
    multistage_df: pd.DataFrame,
    seeds: Sequence[str],
    bootstrap_iterations: int,
    pairwise_rounds: int,
    min_seeds_required: int,
    output_dir: str,
) -> Dict[str, bool]:
    probe_source = _ensure_probe_schema(probe_df)
    seeds_local = list(seeds) if len(seeds) else ["prometheus-2026-p1"]

    if len(multistage_df):
        ms_source = multistage_df.copy()
    else:
        ms_source = pd.DataFrame()

    if len(ms_source):
        for col, default in [
            ("t1_correct", 0),
            ("t3_correct", 0),
            ("degraded", 0),
            ("conf_change", 0.0),
        ]:
            if col not in ms_source.columns:
                ms_source[col] = default
        ms_source["t1_correct"] = ms_source["t1_correct"].map(_norm_bool).astype(int)
        ms_source["t3_correct"] = ms_source["t3_correct"].map(_norm_bool).astype(int)
        ms_source["degraded"] = ms_source["degraded"].map(_norm_bool).astype(int)
        ms_source["conf_change"] = ms_source["conf_change"].map(_norm_float).astype(float)

    parse_quality_summary = (
        probe_source.groupby("model", observed=False)
        .agg(
            n=("problem_id", "count"),
            parse_success_rate=("parse_success", "mean"),
            solvability_present_rate=("solvability_present", "mean"),
            accuracy=("correctness_flag", "mean"),
        )
        .reset_index()
    )
    _save_csv(parse_quality_summary, output_dir, "rg_epoch2_probe_parse_quality_summary.csv")

    probe_group_cols = ["model", "problem_class"]
    if "probe_seed" in probe_source.columns:
        probe_group_cols.append("probe_seed")

    probe_seed_frames: List[pd.DataFrame] = []
    ms_seed_frames: List[pd.DataFrame] = []

    for seed_value in seeds_local:
        probe_parts: List[pd.DataFrame] = []
        for group_name, g in probe_source.groupby(probe_group_cols, observed=False):
            probe_parts.append(
                g.sample(
                    n=len(g),
                    replace=True,
                    random_state=stable_int_seed(f"{seed_value}-probe-{group_name}"),
                )
            )
        seed_probe_df = (
            pd.concat(probe_parts, ignore_index=True)
            if probe_parts
            else probe_source.iloc[0:0].copy()
        )
        seed_probe_df = _ensure_probe_schema(seed_probe_df)
        seed_probe_df["seed"] = seed_value
        probe_seed_frames.append(seed_probe_df)

        if len(ms_source):
            ms_parts: List[pd.DataFrame] = []
            for model_name, g in ms_source.groupby("model", observed=False):
                ms_parts.append(
                    g.sample(
                        n=len(g),
                        replace=True,
                        random_state=stable_int_seed(f"{seed_value}-ms-{model_name}"),
                    )
                )
            seed_ms_df = (
                pd.concat(ms_parts, ignore_index=True)
                if ms_parts
                else ms_source.iloc[0:0].copy()
            )
            seed_ms_df["seed"] = seed_value
            ms_seed_frames.append(seed_ms_df)

    rg_epoch2_probe_seed_items = (
        pd.concat(probe_seed_frames, ignore_index=True)
        if probe_seed_frames
        else pd.DataFrame()
    )
    rg_epoch2_multistage_seed_items = (
        pd.concat(ms_seed_frames, ignore_index=True)
        if ms_seed_frames
        else pd.DataFrame()
    )

    _save_csv(rg_epoch2_probe_seed_items, output_dir, "rg_epoch2_seed_probe_item_level.csv")
    _save_csv(
        rg_epoch2_multistage_seed_items,
        output_dir,
        "rg_epoch2_seed_multistage_item_level.csv",
    )

    if len(rg_epoch2_probe_seed_items) == 0:
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_seed_probe_summary.csv")
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_seed_probe_class_summary.csv")
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_seed_probe_domain_summary.csv")
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_seed_multistage_summary.csv")
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_ci_summary.csv")
        _save_csv(pd.DataFrame(), output_dir, "rg_epoch2_pairwise_significance.csv")
        return {"RG_EPOCH2_MULTI_SEED_PASS": False}

    probe_summary_rows = []
    for (seed_value, model_id), g in rg_epoch2_probe_seed_items.groupby(["seed", "model"], observed=False):
        under = g[g["problem_class"] == "UNDERDETERMINED"]
        contra = g[g["problem_class"] == "CONTRADICTORY"]
        probe_summary_rows.append(
            {
                "seed": seed_value,
                "model": model_id,
                "probe_n": int(len(g)),
                "probe_accuracy": float(g["correctness_flag"].mean()) if len(g) else float("nan"),
                "probe_under_n": int(len(under)),
                "probe_under_accuracy": float(under["correctness_flag"].mean()) if len(under) else float("nan"),
                "probe_contra_n": int(len(contra)),
                "probe_contra_accuracy": float(contra["correctness_flag"].mean()) if len(contra) else float("nan"),
                "probe_parse_success_rate": float(g["parse_success"].mean()) if len(g) else float("nan"),
                "probe_solvability_present_rate": float(g["solvability_present"].mean()) if len(g) else float("nan"),
            }
        )
    probe_summary = pd.DataFrame(probe_summary_rows)

    if len(rg_epoch2_probe_seed_items):
        class_summary = (
            rg_epoch2_probe_seed_items.groupby(["seed", "model", "problem_class"], observed=False)
            .agg(
                n=("problem_id", "count"),
                accuracy=("correctness_flag", "mean"),
                parse_success_rate=("parse_success", "mean"),
                solvability_present_rate=("solvability_present", "mean"),
            )
            .reset_index()
        )
        domain_summary = (
            rg_epoch2_probe_seed_items.groupby(["seed", "model", "domain"], observed=False)
            .agg(
                n=("problem_id", "count"),
                accuracy=("correctness_flag", "mean"),
                parse_success_rate=("parse_success", "mean"),
                solvability_present_rate=("solvability_present", "mean"),
            )
            .reset_index()
        )
    else:
        class_summary = pd.DataFrame()
        domain_summary = pd.DataFrame()

    _save_csv(class_summary, output_dir, "rg_epoch2_seed_probe_class_summary.csv")
    _save_csv(domain_summary, output_dir, "rg_epoch2_seed_probe_domain_summary.csv")

    ms_summary_rows = []
    if len(rg_epoch2_multistage_seed_items):
        for (seed_value, model_id), g in rg_epoch2_multistage_seed_items.groupby(["seed", "model"], observed=False):
            t1_acc = float(g["t1_correct"].mean()) if len(g) else float("nan")
            t3_acc = float(g["t3_correct"].mean()) if len(g) else float("nan")
            ms_summary_rows.append(
                {
                    "seed": seed_value,
                    "model": model_id,
                    "multistage_n": int(len(g)),
                    "multistage_t1_accuracy": t1_acc,
                    "multistage_t3_accuracy": t3_acc,
                    "multistage_delta_accuracy": (
                        (t3_acc - t1_acc)
                        if (not pd.isna(t1_acc) and not pd.isna(t3_acc))
                        else float("nan")
                    ),
                    "multistage_meta_accuracy": float(1.0 - g["degraded"].mean()) if len(g) else float("nan"),
                    "multistage_avg_conf_shift": float(g["conf_change"].mean()) if len(g) else float("nan"),
                }
            )
    ms_summary = pd.DataFrame(ms_summary_rows)

    _save_csv(probe_summary, output_dir, "rg_epoch2_seed_probe_summary.csv")
    _save_csv(ms_summary, output_dir, "rg_epoch2_seed_multistage_summary.csv")

    ci_rows = []
    for model_name, g in probe_summary.groupby("model", observed=False):
        mean_probe, lo_probe, hi_probe = rg_bootstrap_mean_ci(
            g["probe_accuracy"].tolist(),
            n_boot=bootstrap_iterations,
            seed=stable_int_seed(f"epoch2-probe-{model_name}"),
        )
        mean_under, lo_under, hi_under = rg_bootstrap_mean_ci(
            g["probe_under_accuracy"].dropna().tolist(),
            n_boot=bootstrap_iterations,
            seed=stable_int_seed(f"epoch2-under-{model_name}"),
        )
        mean_contra, lo_contra, hi_contra = rg_bootstrap_mean_ci(
            g["probe_contra_accuracy"].dropna().tolist(),
            n_boot=bootstrap_iterations,
            seed=stable_int_seed(f"epoch2-contra-{model_name}"),
        )

        g_ms = ms_summary[ms_summary["model"] == model_name] if len(ms_summary) else pd.DataFrame()
        mean_delta, lo_delta, hi_delta = rg_bootstrap_mean_ci(
            g_ms["multistage_delta_accuracy"].dropna().tolist() if len(g_ms) else [],
            n_boot=bootstrap_iterations,
            seed=stable_int_seed(f"epoch2-msdelta-{model_name}"),
        )

        ci_rows.append(
            {
                "model": model_name,
                "probe_accuracy_mean": mean_probe,
                "probe_accuracy_ci_low": lo_probe,
                "probe_accuracy_ci_high": hi_probe,
                "probe_under_accuracy_mean": mean_under,
                "probe_under_accuracy_ci_low": lo_under,
                "probe_under_accuracy_ci_high": hi_under,
                "probe_contra_accuracy_mean": mean_contra,
                "probe_contra_accuracy_ci_low": lo_contra,
                "probe_contra_accuracy_ci_high": hi_contra,
                "multistage_delta_mean": mean_delta,
                "multistage_delta_ci_low": lo_delta,
                "multistage_delta_ci_high": hi_delta,
                "n_probe_seeds": int(len(g)),
                "n_multistage_seeds": int(len(g_ms)),
            }
        )

    rg_epoch2_ci = (
        pd.DataFrame(ci_rows).sort_values("probe_accuracy_mean", ascending=False)
        if ci_rows
        else pd.DataFrame()
    )
    _save_csv(rg_epoch2_ci, output_dir, "rg_epoch2_ci_summary.csv")

    pair_rows = []
    unique_models = sorted(probe_summary["model"].unique().tolist()) if len(probe_summary) else []
    for a, b in itertools.combinations(unique_models, 2):
        a_vals = probe_summary[probe_summary["model"] == a]["probe_accuracy"].tolist()
        b_vals = probe_summary[probe_summary["model"] == b]["probe_accuracy"].tolist()
        p_probe = rg_permutation_pvalue(
            a_vals,
            b_vals,
            rounds=pairwise_rounds,
            seed=stable_int_seed(f"epoch2|probe|{a}|{b}"),
        )
        h_probe = rg_cohens_h(statistics.mean(a_vals), statistics.mean(b_vals))

        a_ms_vals = (
            ms_summary[ms_summary["model"] == a]["multistage_delta_accuracy"].dropna().tolist()
            if len(ms_summary)
            else []
        )
        b_ms_vals = (
            ms_summary[ms_summary["model"] == b]["multistage_delta_accuracy"].dropna().tolist()
            if len(ms_summary)
            else []
        )
        if len(a_ms_vals) > 0 and len(b_ms_vals) > 0:
            p_ms = rg_permutation_pvalue(
                a_ms_vals,
                b_ms_vals,
                rounds=pairwise_rounds,
                seed=stable_int_seed(f"epoch2|ms|{a}|{b}"),
            )
            h_ms = rg_cohens_h(statistics.mean(a_ms_vals), statistics.mean(b_ms_vals))
        else:
            p_ms = float("nan")
            h_ms = float("nan")

        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "probe_mean_a": statistics.mean(a_vals),
                "probe_mean_b": statistics.mean(b_vals),
                "probe_delta_a_minus_b": statistics.mean(a_vals) - statistics.mean(b_vals),
                "probe_permutation_pvalue": p_probe,
                "probe_cohens_h": h_probe,
                "probe_effect_size": rg_effect_label(h_probe),
                "multistage_delta_mean_a": statistics.mean(a_ms_vals) if len(a_ms_vals) else float("nan"),
                "multistage_delta_mean_b": statistics.mean(b_ms_vals) if len(b_ms_vals) else float("nan"),
                "multistage_delta_permutation_pvalue": p_ms,
                "multistage_delta_cohens_h": h_ms,
                "multistage_delta_effect_size": rg_effect_label(h_ms) if not pd.isna(h_ms) else "na",
            }
        )

    rg_epoch2_pairwise = (
        pd.DataFrame(pair_rows).sort_values("probe_permutation_pvalue")
        if pair_rows
        else pd.DataFrame()
    )
    _save_csv(rg_epoch2_pairwise, output_dir, "rg_epoch2_pairwise_significance.csv")

    probe_seed_counts = (
        probe_summary.groupby("model", observed=False)["seed"].nunique()
        if len(probe_summary)
        else pd.Series(dtype=int)
    )
    probe_seed_ok = len(probe_seed_counts) > 0 and int(probe_seed_counts.min()) >= min_seeds_required

    if len(ms_summary):
        ms_seed_counts = ms_summary.groupby("model", observed=False)["seed"].nunique()
        ms_seed_ok = len(ms_seed_counts) > 0 and int(ms_seed_counts.min()) >= min_seeds_required
    else:
        ms_seed_ok = False

    return {"RG_EPOCH2_MULTI_SEED_PASS": bool(probe_seed_ok and ms_seed_ok)}


def write_contamination_audit(
    base_source: Sequence[Dict],
    probe_source: Sequence[Dict],
    output_dir: str,
) -> Dict[str, bool]:
    import re

    def _norm_text(text: object) -> str:
        s = str(text or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _token_set(text: object) -> set:
        return set(_norm_text(text).split())

    base_ids = {str(x.get("problem_id", "")) for x in base_source}
    probe_ids = {str(x.get("problem_id", "")) for x in probe_source}
    id_overlap = sorted(base_ids.intersection(probe_ids))

    base_text_map: Dict[str, List[str]] = {}
    for x in base_source:
        key = _norm_text(x.get("user", x.get("question", "")))
        if key:
            base_text_map.setdefault(key, []).append(str(x.get("problem_id", "")))

    probe_text_map: Dict[str, List[str]] = {}
    for x in probe_source:
        key = _norm_text(x.get("user", x.get("question", "")))
        if key:
            probe_text_map.setdefault(key, []).append(str(x.get("problem_id", "")))

    exact_text_overlap = []
    for key in sorted(set(base_text_map.keys()).intersection(probe_text_map.keys())):
        exact_text_overlap.append(
            {
                "normalized_prompt": key,
                "base_ids": base_text_map[key],
                "probe_ids": probe_text_map[key],
            }
        )

    near_pairs = []
    base_samples = [
        (str(x.get("problem_id", "")), _token_set(x.get("user", x.get("question", ""))))
        for x in base_source
    ]
    probe_samples = [
        (str(x.get("problem_id", "")), _token_set(x.get("user", x.get("question", ""))))
        for x in probe_source
    ]

    for probe_id, probe_tokens in probe_samples:
        if not probe_tokens:
            continue
        for base_id, base_tokens in base_samples:
            if not base_tokens:
                continue
            inter = len(probe_tokens.intersection(base_tokens))
            union = len(probe_tokens.union(base_tokens))
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= 0.90:
                near_pairs.append(
                    {
                        "base_problem_id": base_id,
                        "probe_problem_id": probe_id,
                        "jaccard": round(jaccard, 4),
                    }
                )

    near_pairs = sorted(
        near_pairs,
        key=lambda x: (-x["jaccard"], x["probe_problem_id"], x["base_problem_id"]),
    )

    contamination_report = {
        "generated_at_utc": utc_now_iso(),
        "base_count": len(base_source),
        "probe_count": len(probe_source),
        "id_overlap_count": len(id_overlap),
        "exact_text_overlap_count": len(exact_text_overlap),
        "near_duplicate_count_jaccard_ge_0_90": len(near_pairs),
        "id_overlap_ids": id_overlap[:20],
    }

    _save_json(contamination_report, output_dir, "contamination_audit_report.json")
    _save_csv(pd.DataFrame(near_pairs), output_dir, "contamination_overlap_pairs.csv")

    contamination_pass = bool(
        contamination_report["id_overlap_count"] == 0
        and contamination_report["exact_text_overlap_count"] == 0
        and contamination_report["near_duplicate_count_jaccard_ge_0_90"] == 0
    )
    return {"CONTAMINATION_AUDIT_PASS": contamination_pass}


def resolve_judges(
    candidates: Sequence[str],
    available: Sequence[str],
) -> List[str]:
    available_list = [str(x) for x in available]
    available_lower = {x.lower(): x for x in available_list}

    resolved: List[str] = []
    seen = set()

    for target in candidates:
        t = str(target).lower()
        if t in available_lower:
            name = available_lower[t]
            if name not in seen:
                resolved.append(name)
                seen.add(name)
            continue

        partial = [n for n in available_list if t in n.lower() or n.lower() in t]
        if partial:
            name = sorted(partial, key=len)[0]
            if name not in seen:
                resolved.append(name)
                seen.add(name)

    if len(resolved) < 2:
        for name in available_list:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
            if len(resolved) >= 3:
                break

    return resolved


def write_judge_sensitivity_artifacts(
    item_df: pd.DataFrame,
    providers,
    candidates: Sequence[str],
    sample_max: int,
    disagreement_threshold: float,
    enabled: bool,
    output_dir: str,
    verbose: bool = False,
) -> Dict[str, bool]:
    if not enabled:
        report = {
            "generated_at_utc": utc_now_iso(),
            "status": "skipped_by_config",
            "reason": "run_independent_judge_sensitivity=False",
            "resolved_judges": [],
            "pass": False,
        }
        _save_json(report, output_dir, "independent_judge_sensitivity_report.json")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_item_eval.csv")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_pairwise_disagreement.csv")
        return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": False}

    if providers is None or not hasattr(providers, "prompt"):
        report = {
            "generated_at_utc": utc_now_iso(),
            "status": "failed",
            "reason": "provider_missing_prompt_interface",
            "resolved_judges": [],
            "pass": False,
        }
        _save_json(report, output_dir, "independent_judge_sensitivity_report.json")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_item_eval.csv")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_pairwise_disagreement.csv")
        return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": False}

    if len(item_df) == 0:
        report = {
            "generated_at_utc": utc_now_iso(),
            "status": "failed",
            "reason": "empty_item_level_data",
            "resolved_judges": [],
            "pass": False,
        }
        _save_json(report, output_dir, "independent_judge_sensitivity_report.json")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_item_eval.csv")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_pairwise_disagreement.csv")
        return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": False}

    available = []
    if hasattr(providers, "list_models"):
        try:
            available = list(providers.list_models())
        except Exception:
            available = []

    if not available:
        # Fallback for providers that cannot list models: trust candidates directly.
        available = list(candidates)

    resolved_judges = resolve_judges(candidates, available)
    if len(resolved_judges) < 2:
        report = {
            "generated_at_utc": utc_now_iso(),
            "status": "failed",
            "reason": "fewer_than_two_judges_resolved",
            "resolved_judges": resolved_judges,
            "pass": False,
        }
        _save_json(report, output_dir, "independent_judge_sensitivity_report.json")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_item_eval.csv")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_pairwise_disagreement.csv")
        return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": False}

    source = item_df.copy().reset_index(drop=True)
    source["sample_row_id"] = source.index.astype(int)

    if len(source) > sample_max:
        rng = np.random.default_rng(stable_int_seed("judge-sensitivity-sampling"))
        picked_ids: List[int] = []

        by_model = list(source.groupby("model", observed=False)) if "model" in source.columns else [("all", source)]
        per_model = max(1, sample_max // max(1, len(by_model)))

        for _, g in by_model:
            take = min(len(g), per_model)
            if take > 0:
                chosen = rng.choice(g["sample_row_id"].to_numpy(), size=take, replace=False)
                picked_ids.extend([int(x) for x in chosen])

        if len(set(picked_ids)) < sample_max:
            remaining = sorted(set(source["sample_row_id"].tolist()) - set(picked_ids))
            need = sample_max - len(set(picked_ids))
            if need > 0 and remaining:
                add = rng.choice(np.array(remaining, dtype=int), size=min(need, len(remaining)), replace=False)
                picked_ids.extend([int(x) for x in add])

        picked_ids = sorted(set(picked_ids))[:sample_max]
        sample_df = source[source["sample_row_id"].isin(picked_ids)].copy().reset_index(drop=True)
    else:
        sample_df = source.copy().reset_index(drop=True)

    judge_rows: List[Dict] = []
    for judge_name in resolved_judges:
        if verbose:
            print(f"[RG][JudgeSensitivity] Running judge: {judge_name}")

        for _, row in sample_df.iterrows():
            gt = row.get("ground_truth", None)
            answer = row.get("final_answer", None)
            if pd.isna(gt) or str(gt).strip() in {"", "None", "nan"}:
                continue

            judge_prompt = (
                "Is the following model answer correct given the ground truth?\n\n"
                f"Ground truth: {gt}\n"
                f"Model answer: {answer}\n\n"
                "Reply with only: CORRECT or INCORRECT"
            )

            try:
                resp = providers.prompt(judge_name, "You are an expert grade evaluator.", judge_prompt)
                text = str(resp).upper()
                if "INCORRECT" in text:
                    is_correct = 0
                elif "CORRECT" in text:
                    is_correct = 1
                else:
                    continue
            except Exception:
                continue

            judge_rows.append(
                {
                    "sample_row_id": int(row["sample_row_id"]),
                    "judge_model": judge_name,
                    "evaluated_model": row.get("model", ""),
                    "problem_id": row.get("problem_id", ""),
                    "problem_class": row.get("problem_class", ""),
                    "solvability_class": row.get("solvability_class", row.get("predicted_class", "")),
                    "is_correct_under_judge": int(is_correct),
                    "evaluation_method": "judge_model",
                }
            )

    judge_eval_df = pd.DataFrame(judge_rows)
    _save_csv(judge_eval_df, output_dir, "independent_judge_item_eval.csv")

    if len(judge_eval_df) == 0:
        report = {
            "generated_at_utc": utc_now_iso(),
            "status": "failed",
            "reason": "judge_calls_failed_or_empty",
            "resolved_judges": resolved_judges,
            "sample_rows": int(len(sample_df)),
            "pass": False,
        }
        _save_json(report, output_dir, "independent_judge_sensitivity_report.json")
        _save_csv(pd.DataFrame(), output_dir, "independent_judge_pairwise_disagreement.csv")
        return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": False}

    wide = judge_eval_df.pivot_table(
        index="sample_row_id",
        columns="judge_model",
        values="is_correct_under_judge",
        aggfunc="first",
    )

    pair_rows = []
    for a, b in itertools.combinations(resolved_judges, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        mask = wide[a].notna() & wide[b].notna()
        n_overlap = int(mask.sum())
        if n_overlap == 0:
            continue
        disagree = float((wide.loc[mask, a] != wide.loc[mask, b]).mean())
        pair_rows.append(
            {
                "judge_a": a,
                "judge_b": b,
                "n_overlap": n_overlap,
                "disagreement_rate": disagree,
                "agreement_rate": 1.0 - disagree,
            }
        )

    pair_df = pd.DataFrame(pair_rows).sort_values("disagreement_rate", ascending=False) if pair_rows else pd.DataFrame()
    _save_csv(pair_df, output_dir, "independent_judge_pairwise_disagreement.csv")

    max_disagree = float(pair_df["disagreement_rate"].max()) if len(pair_df) > 0 else float("nan")
    pass_flag = bool(
        len(resolved_judges) >= 2
        and len(judge_eval_df) > 0
        and len(pair_df) > 0
        and (not pd.isna(max_disagree))
        and max_disagree <= disagreement_threshold
    )

    sensitivity_report = {
        "generated_at_utc": utc_now_iso(),
        "resolved_judges": resolved_judges,
        "sample_rows": int(len(sample_df)),
        "judge_evaluations": int(len(judge_eval_df)),
        "pairwise_comparisons": int(len(pair_df)),
        "max_disagreement_rate": None if pd.isna(max_disagree) else max_disagree,
        "disagreement_threshold": float(disagreement_threshold),
        "pass": bool(pass_flag),
    }
    _save_json(sensitivity_report, output_dir, "independent_judge_sensitivity_report.json")
    return {"INDEPENDENT_JUDGE_SENSITIVITY_PASS": pass_flag}


def write_epoch1_bundle(
    item_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, object]:
    _save_csv(item_df, output_dir, "prometheus_item_level_results.csv")
    _save_csv(summary_df, output_dir, "prometheus_model_comparison.csv")

    item_json_path = os.path.join(output_dir, "prometheus_item_level_results.json")
    summary_json_path = os.path.join(output_dir, "prometheus_model_comparison.json")

    item_df.to_json(item_json_path, orient="records", indent=2)
    summary_df.to_json(summary_json_path, orient="records", indent=2)

    run_meta = {
        "generated_at_utc": utc_now_iso(),
        "item_rows": int(len(item_df)),
        "summary_rows": int(len(summary_df)),
        "models": sorted(summary_df["model"].astype(str).tolist()) if "model" in summary_df.columns else [],
        "notes": "Generated by prometheus_ebm SDK export pipeline.",
    }
    _save_json(run_meta, output_dir, "prometheus_export_manifest.json")

    bundle_files = [
        "prometheus_item_level_results.csv",
        "prometheus_model_comparison.csv",
        "prometheus_item_level_results.json",
        "prometheus_model_comparison.json",
        "prometheus_export_manifest.json",
    ]

    zip_name = os.path.join(output_dir, "prometheus_results_export.zip")
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in bundle_files:
            zf.write(os.path.join(output_dir, file_name), arcname=file_name)

    return {"manifest": run_meta, "zip_path": zip_name}


def write_epoch2_bundle(
    probe_df: pd.DataFrame,
    multistage_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, object]:
    probe_norm = _ensure_probe_schema(probe_df)
    _save_csv(probe_norm, output_dir, "probe_results.csv")

    class_breakdown = (
        probe_norm.groupby(["model", "problem_class"], observed=False)
        .agg(
            n=("problem_id", "count"),
            accuracy=("correctness_flag", "mean"),
            mean_confidence=("confidence", "mean"),
            parse_success_rate=("parse_success", "mean"),
            solvability_present_rate=("solvability_present", "mean"),
        )
        .reset_index()
    )
    _save_csv(class_breakdown, output_dir, "probe_class_breakdown.csv")

    model_metrics = (
        probe_norm.groupby("model", observed=False)
        .agg(
            n=("problem_id", "count"),
            probe_seed_count=("probe_seed", "nunique") if "probe_seed" in probe_norm.columns else ("problem_id", "count"),
            accuracy=("correctness_flag", "mean"),
            mean_confidence=("confidence", "mean"),
            parse_success_rate=("parse_success", "mean"),
            solvability_present_rate=("solvability_present", "mean"),
        )
        .reset_index()
    )
    _save_csv(model_metrics, output_dir, "probe_model_detailed_metrics.csv")

    parse_route_report = (
        probe_norm.groupby(["model", "parse_route"], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
        if "parse_route" in probe_norm.columns
        else pd.DataFrame()
    )
    _save_csv(parse_route_report, output_dir, "probe_parse_quality_report.csv")

    quality_manifest = {
        "probe_rows": int(len(probe_norm)),
        "models": sorted(probe_norm["model"].astype(str).unique().tolist()),
        "overall_accuracy": float(probe_norm["correctness_flag"].mean()) if len(probe_norm) else 0.0,
        "overall_parse_success_rate": float(probe_norm["parse_success"].mean()) if len(probe_norm) else 0.0,
        "overall_solvability_present_rate": float(probe_norm["solvability_present"].mean()) if len(probe_norm) else 0.0,
    }
    _save_json(quality_manifest, output_dir, "probe_quality_manifest.json")

    ms_df = multistage_df.copy() if len(multistage_df) else pd.DataFrame()
    _save_csv(ms_df, output_dir, "multistage_results.csv")

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "epoch": 2,
        "artifact_boundary": "epoch2_only",
        "contains_epoch1_files": False,
        "probe_results_rows": int(len(probe_norm)),
        "multistage_results_rows": int(len(ms_df)),
        "models": sorted(probe_norm["model"].astype(str).unique().tolist()),
    }
    _save_json(manifest, output_dir, "epoch2_manifest.json")

    files_to_zip = [
        "probe_results.csv",
        "multistage_results.csv",
        "epoch2_manifest.json",
        "probe_class_breakdown.csv",
        "probe_model_detailed_metrics.csv",
        "probe_parse_quality_report.csv",
        "probe_quality_manifest.json",
    ]

    epoch1_like_files = {
        "prometheus_item_level_results.csv",
        "prometheus_model_comparison.csv",
        "prometheus_item_level_results.json",
        "prometheus_model_comparison.json",
        "prometheus_export_manifest.json",
        "prometheus_results_export.zip",
    }

    boundary_collision = sorted(set(files_to_zip).intersection(epoch1_like_files))
    if boundary_collision:
        raise RuntimeError(f"Epoch boundary violation before zip write: {boundary_collision}")

    zip_name = os.path.join(output_dir, "prometheus_epoch2_export.zip")
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_zip:
            full = os.path.join(output_dir, fname)
            if os.path.exists(full):
                zf.write(full, arcname=fname)

    return {"manifest": manifest, "zip_path": zip_name}


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _json_file(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _zip_members(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return set(zf.namelist())
    except Exception:
        return set()


def write_gate_and_card(
    output_dir: str,
    rg_epoch1_multi_seed_pass: bool,
    rg_epoch2_multi_seed_pass: bool,
    contamination_pass: bool,
    judge_pass: bool,
) -> Dict[str, object]:
    epoch1_ci_df = _safe_read_csv(os.path.join(output_dir, "rg_epoch1_eci_hgi_ci.csv"))
    epoch1_sig_df = _safe_read_csv(os.path.join(output_dir, "rg_epoch1_pairwise_significance.csv"))
    epoch2_ci_df = _safe_read_csv(os.path.join(output_dir, "rg_epoch2_ci_summary.csv"))
    epoch2_sig_df = _safe_read_csv(os.path.join(output_dir, "rg_epoch2_pairwise_significance.csv"))

    contamination_report = _json_file(os.path.join(output_dir, "contamination_audit_report.json"))
    judge_sensitivity_report = _json_file(os.path.join(output_dir, "independent_judge_sensitivity_report.json"))

    epoch1_manifest = _json_file(os.path.join(output_dir, "prometheus_export_manifest.json"))
    epoch2_manifest = _json_file(os.path.join(output_dir, "epoch2_manifest.json"))

    epoch1_zip_members = _zip_members(os.path.join(output_dir, "prometheus_results_export.zip"))
    epoch2_zip_members = _zip_members(os.path.join(output_dir, "prometheus_epoch2_export.zip"))

    epoch1_expected_core = {
        "prometheus_item_level_results.csv",
        "prometheus_model_comparison.csv",
        "prometheus_item_level_results.json",
        "prometheus_model_comparison.json",
        "prometheus_export_manifest.json",
    }
    epoch2_expected_core = {
        "probe_results.csv",
        "multistage_results.csv",
        "epoch2_manifest.json",
    }

    epoch1_forbidden = epoch2_expected_core.union({"prometheus_epoch2_export.zip"})
    epoch2_forbidden = epoch1_expected_core.union({"prometheus_results_export.zip"})

    epoch1_boundary_ok = (
        len(epoch1_zip_members) > 0
        and len(epoch1_zip_members.intersection(epoch1_expected_core)) >= 2
        and len(epoch1_zip_members.intersection(epoch1_forbidden)) == 0
    )
    epoch2_boundary_ok = (
        len(epoch2_zip_members) > 0
        and len(epoch2_zip_members.intersection(epoch2_expected_core)) >= 2
        and len(epoch2_zip_members.intersection(epoch2_forbidden)) == 0
    )

    epoch1_manifest_ok = bool(epoch1_manifest is not None and "generated_at_utc" in epoch1_manifest)
    epoch2_manifest_ok = bool(
        epoch2_manifest is not None
        and int(epoch2_manifest.get("epoch", -1)) == 2
        and epoch2_manifest.get("artifact_boundary") == "epoch2_only"
    )

    criterion_1_multi_seed = bool(rg_epoch1_multi_seed_pass and rg_epoch2_multi_seed_pass)
    criterion_2_ci_sig = bool(
        len(epoch1_ci_df) > 0
        and len(epoch1_sig_df) > 0
        and len(epoch2_ci_df) > 0
        and len(epoch2_sig_df) > 0
    )

    if contamination_report is not None:
        criterion_3_contamination = bool(
            contamination_pass
            and int(contamination_report.get("id_overlap_count", 1)) == 0
            and int(contamination_report.get("exact_text_overlap_count", 1)) == 0
            and int(contamination_report.get("near_duplicate_count_jaccard_ge_0_90", 1)) == 0
        )
    else:
        criterion_3_contamination = bool(contamination_pass)

    if judge_sensitivity_report is not None:
        criterion_4_judge = bool(judge_pass and judge_sensitivity_report.get("pass", False))
    else:
        criterion_4_judge = bool(judge_pass)

    criterion_5_epoch_boundary = bool(
        epoch1_manifest_ok and epoch2_manifest_ok and epoch1_boundary_ok and epoch2_boundary_ok
    )

    top_epoch1_line = "n/a"
    if len(epoch1_ci_df) > 0 and "eci_mean" in epoch1_ci_df.columns:
        e1 = epoch1_ci_df.sort_values("eci_mean", ascending=False).iloc[0]
        top_epoch1_line = (
            f"{e1['model']} | ECI={float(e1['eci_mean']):.4f} "
            f"(95% CI {float(e1['eci_ci_low']):.4f} to {float(e1['eci_ci_high']):.4f})"
        )

    top_epoch2_line = "n/a"
    if len(epoch2_ci_df) > 0 and "probe_accuracy_mean" in epoch2_ci_df.columns:
        e2 = epoch2_ci_df.sort_values("probe_accuracy_mean", ascending=False).iloc[0]
        top_epoch2_line = (
            f"{e2['model']} | ProbeAcc={float(e2['probe_accuracy_mean']):.4f} "
            f"(95% CI {float(e2['probe_accuracy_ci_low']):.4f} to {float(e2['probe_accuracy_ci_high']):.4f})"
        )

    card_path = os.path.join(output_dir, "benchmark_card_research_grade_v1.md")
    card_lines = [
        "# PROMETHEUS Benchmark Card (Research-Grade v1)",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Protocol Summary",
        "- Epoch-1 and Epoch-2 are evaluated once, then analyzed with deterministic multi-seed bootstrap resampling (>=2 seeds).",
        "- Core claims reported with bootstrap 95% confidence intervals and pairwise permutation tests over resampled distributions.",
        "- Probe contamination and leakage audited with ID/text overlap checks.",
        "- Independent-judge sensitivity measured on sampled item-level outputs.",
        "- Artifact boundaries validated across Epoch-1 and Epoch-2 bundles.",
        "",
        "## Core Results Snapshot",
        f"- Epoch-1 top model: {top_epoch1_line}",
        f"- Epoch-2 top model: {top_epoch2_line}",
        "",
        "## Artifacts",
        "- rg_epoch1_seed_summary.csv",
        "- rg_epoch1_eci_hgi_ci.csv",
        "- rg_epoch1_component_ci.csv",
        "- rg_epoch1_seed_class_summary.csv",
        "- rg_epoch1_seed_domain_summary.csv",
        "- rg_epoch1_pairwise_significance.csv",
        "- rg_epoch1_pairwise_significance_hgi.csv",
        "- contamination_audit_report.json",
        "- independent_judge_sensitivity_report.json",
        "- rg_epoch2_probe_parse_quality_summary.csv",
        "- rg_epoch2_seed_probe_summary.csv",
        "- rg_epoch2_seed_probe_class_summary.csv",
        "- rg_epoch2_seed_probe_domain_summary.csv",
        "- rg_epoch2_seed_multistage_summary.csv",
        "- rg_epoch2_ci_summary.csv",
        "- rg_epoch2_pairwise_significance.csv",
        "- prometheus_results_export.zip",
        "- prometheus_epoch2_export.zip",
        "",
        "## Six-Criterion Gate Status",
        f"- C1 Multi-seed both epochs: {criterion_1_multi_seed}",
        f"- C2 CI and significance artifacts: {criterion_2_ci_sig}",
        f"- C3 Contamination audit clean: {criterion_3_contamination}",
        f"- C4 Independent-judge sensitivity pass: {criterion_4_judge}",
        f"- C5 Strict epoch boundaries: {criterion_5_epoch_boundary}",
    ]

    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines) + "\n")

    criterion_6_card = bool(os.path.exists(card_path) and os.path.getsize(card_path) > 0)

    criteria = {
        "1_multi_seed_both_epochs": criterion_1_multi_seed,
        "2_ci_significance_for_core_claims": criterion_2_ci_sig,
        "3_contamination_audit_clean": criterion_3_contamination,
        "4_independent_judge_sensitivity": criterion_4_judge,
        "5_strict_epoch_boundaries": criterion_5_epoch_boundary,
        "6_benchmark_card_generated": criterion_6_card,
    }

    all_validation_criteria_met = bool(all(criteria.values()))

    gate_payload = {
        "generated_at_utc": utc_now_iso(),
        "claim_research_grade_v1": all_validation_criteria_met,
        "criteria": criteria,
        "evidence_files": {
            "epoch1_ci": "rg_epoch1_eci_hgi_ci.csv",
            "epoch1_component_ci": "rg_epoch1_component_ci.csv",
            "epoch1_pairwise_significance": "rg_epoch1_pairwise_significance.csv",
            "epoch1_pairwise_hgi": "rg_epoch1_pairwise_significance_hgi.csv",
            "contamination_report": "contamination_audit_report.json",
            "judge_sensitivity": "independent_judge_sensitivity_report.json",
            "epoch2_ci": "rg_epoch2_ci_summary.csv",
            "epoch2_pairwise_significance": "rg_epoch2_pairwise_significance.csv",
            "epoch2_parse_quality": "rg_epoch2_probe_parse_quality_summary.csv",
            "epoch2_class_summary": "rg_epoch2_seed_probe_class_summary.csv",
            "epoch2_domain_summary": "rg_epoch2_seed_probe_domain_summary.csv",
            "epoch1_bundle": "prometheus_results_export.zip",
            "epoch2_bundle": "prometheus_epoch2_export.zip",
            "benchmark_card": "benchmark_card_research_grade_v1.md",
        },
        "boundary_checks": {
            "epoch1_manifest_ok": epoch1_manifest_ok,
            "epoch2_manifest_ok": epoch2_manifest_ok,
            "epoch1_zip_boundary_ok": epoch1_boundary_ok,
            "epoch2_zip_boundary_ok": epoch2_boundary_ok,
        },
    }

    _save_json(gate_payload, output_dir, "research_grade_v1_gate.json")
    criteria_df = pd.DataFrame([{"criterion": k, "pass": bool(v)} for k, v in criteria.items()])
    _save_csv(criteria_df, output_dir, "research_grade_v1_gate_criteria.csv")

    return {
        "RESEARCH_GRADE_V1_PASS": all_validation_criteria_met,
        "criteria": criteria,
    }


def write_master_bundle(output_dir: str, master_bundle_name: str) -> str:
    all_files = [
        "prometheus_model_comparison.csv",
        "prometheus_model_comparison.json",
        "prometheus_item_level_results.csv",
        "prometheus_item_level_results.json",
        "prometheus_export_manifest.json",
        "probe_results.csv",
        "probe_class_breakdown.csv",
        "probe_model_detailed_metrics.csv",
        "probe_parse_quality_report.csv",
        "probe_quality_manifest.json",
        "multistage_results.csv",
        "epoch2_manifest.json",
        "rg_epoch1_seed_summary.csv",
        "rg_epoch1_seed_item_level.csv",
        "rg_epoch1_seed_class_summary.csv",
        "rg_epoch1_seed_domain_summary.csv",
        "rg_epoch1_eci_hgi_ci.csv",
        "rg_epoch1_component_ci.csv",
        "rg_epoch1_pairwise_significance.csv",
        "rg_epoch1_pairwise_significance_hgi.csv",
        "rg_epoch2_ci_summary.csv",
        "rg_epoch2_pairwise_significance.csv",
        "rg_epoch2_probe_parse_quality_summary.csv",
        "rg_epoch2_seed_probe_summary.csv",
        "rg_epoch2_seed_probe_class_summary.csv",
        "rg_epoch2_seed_probe_domain_summary.csv",
        "rg_epoch2_seed_probe_item_level.csv",
        "rg_epoch2_seed_multistage_summary.csv",
        "rg_epoch2_seed_multistage_item_level.csv",
        "contamination_audit_report.json",
        "contamination_overlap_pairs.csv",
        "independent_judge_sensitivity_report.json",
        "independent_judge_item_eval.csv",
        "independent_judge_pairwise_disagreement.csv",
        "research_grade_v1_gate.json",
        "research_grade_v1_gate_criteria.csv",
        "prometheus_brier_dprime.csv",
        "benchmark_card_research_grade_v1.md",
        "prometheus_results_export.zip",
        "prometheus_epoch2_export.zip",
        "epistemic_radar.png",
        "reliability_diagram.png",
        "edki_scatter.png",
    ]

    found = [f for f in all_files if os.path.exists(os.path.join(output_dir, f))]

    bundle_path = os.path.join(output_dir, master_bundle_name)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in found:
            zf.write(os.path.join(output_dir, file_name), arcname=file_name)

    return bundle_path
