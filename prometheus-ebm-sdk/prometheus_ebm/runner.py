"""PROMETHEUS-EBM Runner.

This runner executes benchmark evaluation and can export a full V5-compatible
artifact bundle for independent lab replication.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import RunConfig
from .research_grade import (
    write_contamination_audit,
    write_epoch1_artifacts,
    write_epoch1_bundle,
    write_epoch2_artifacts,
    write_epoch2_bundle,
    write_gate_and_card,
    write_judge_sensitivity_artifacts,
    write_master_bundle,
)
from .scorer import BrierDecomposition, ECIScorer, ScoringResult, Type2DPrime


class PrometheusRunner:
    """Main benchmark runner for PROMETHEUS-EBM."""

    def __init__(self, config: RunConfig, provider=None):
        self.config = config
        self.config.validate()
        self.scorer = ECIScorer()
        self.results: Optional["BenchmarkResults"] = None
        self._start_time: Optional[float] = None
        self._provider_override = provider

        self._base_dataset: List[Dict] = []
        self._probe_source: List[Dict] = []
        self._providers = None
        self._multistage_dataframe = pd.DataFrame()

    def run(self) -> "BenchmarkResults":
        """Execute the full benchmark pipeline."""
        self._start_time = time.time()

        if self.config.verbose:
            print(self.config.summary())

        # 1. Load dataset (mode-aware defaults).
        base_dataset = self._load_dataset()
        self._base_dataset = [dict(p) for p in base_dataset]

        # 2. Apply stress augmentation.
        dataset = self._augment_dataset(base_dataset)

        # 3. Resolve model providers.
        providers = self._provider_override if self._provider_override is not None else self._resolve_providers()
        self._providers = providers

        # 4. Run evaluation loop.
        raw_results = self._evaluate(dataset, providers)

        # 4.3 Run multi-stage protocol on a sampled subset (separate from base scoring).
        multistage_results = pd.DataFrame()
        if self.config.run_multistage:
            multistage_results = self._evaluate_multistage(base_dataset, providers)
        self._multistage_dataframe = multistage_results

        # 4.5 Run probes.
        probe_results, probe_source = self._evaluate_probes(providers)
        self._probe_source = probe_source

        # 5. Score results.
        scored = self._score(raw_results)

        # 6. Statistical validation.
        if self.config.run_statistics:
            stats = self._bootstrap(raw_results, scored)
        else:
            stats = None

        self.results = BenchmarkResults(
            config=self.config,
            model_scores=scored,
            raw_dataframe=raw_results,
            probe_dataframe=probe_results,
            multistage_dataframe=multistage_results,
            statistics=stats,
            elapsed_seconds=time.time() - self._start_time,
            base_dataset=self._base_dataset,
            probe_source=self._probe_source,
            providers=self._providers,
        )
        return self.results

    def run_all(self) -> "BenchmarkResults":
        """Compatibility alias used by older examples and lab scripts."""
        return self.run()

    def _load_dataset(self) -> List[Dict]:
        """Load the base problem set with mode-aware defaults."""
        path = self.config.dataset_path
        pkg_dir = os.path.dirname(os.path.abspath(__file__))

        if path is None:
            if self.config.normalized_mode() == "deep_probe":
                default_name = "prometheus_1000_dataset.json"
            else:
                default_name = "prometheus_200_multimodel_dataset.json"
            path = os.path.join(pkg_dir, "data", default_name)
        elif not os.path.isabs(path):
            # Resolve relative paths against CWD first, then bundled data folder.
            if os.path.exists(path):
                path = os.path.abspath(path)
            else:
                bundled = os.path.join(pkg_dir, "data", path)
                if os.path.exists(bundled):
                    path = bundled

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Download from: https://github.com/Mushfiqul-Alam-17/prometheus-ebm-sdk"
            )

        with open(path, "r", encoding="utf-8") as f:
            problems = json.load(f)

        if self.config.n_items < len(problems):
            problems = problems[: self.config.n_items]

        if self.config.verbose:
            print(f"  Loaded {len(problems)} base problems from {path}")

        return problems

    def _augment_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Apply stress augmentation to create challenge variants."""
        import random

        augmented = list(dataset)
        rng = random.Random(self.config.seeds[0] if self.config.seeds else 42)

        for p in dataset:
            roll = rng.random()
            if roll < self.config.stress_decision_ratio:
                aug = dict(p)
                aug["problem_id"] = p["problem_id"] + "-DS"
                base_text = p.get("user", p.get("question", ""))
                aug["user"] = (
                    base_text
                    + "\n\nAdditional instruction: "
                    + "Before finalizing your answer, explicitly test at least one plausible "
                    + "alternative interpretation and reject it with evidence if unsupported."
                )
                aug["rigor_mode"] = "decision_stress"
                augmented.append(aug)
            elif roll < (self.config.stress_decision_ratio + self.config.stress_clarity_ratio):
                aug = dict(p)
                aug["problem_id"] = p["problem_id"] + "-CS"
                base_text = p.get("user", p.get("question", ""))
                aug["user"] = (
                    base_text
                    + "\n\nAdditional instruction: "
                    + "Maintain strict adherence to the response schema. Avoid any speculative "
                    + "claims not directly supported by the information given."
                )
                aug["rigor_mode"] = "clarity_stress"
                augmented.append(aug)

        if self.config.verbose:
            print(f"  Stress augmentation: {len(dataset)} base -> {len(augmented)} total prompts")
        return augmented

    def _resolve_providers(self):
        """Resolve model API providers based on config."""
        provider_name = self.config.provider

        if provider_name == "kaggle":
            from prometheus_ebm.providers.kaggle import KaggleProvider

            return KaggleProvider(self.config)
        if provider_name == "openrouter":
            from prometheus_ebm.providers.openrouter import OpenRouterProvider

            return OpenRouterProvider(self.config)
        if provider_name == "anthropic":
            from prometheus_ebm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(self.config)
        if provider_name == "openai":
            from prometheus_ebm.providers.openai import OpenAIProvider

            return OpenAIProvider(self.config)

        raise ValueError(f"Unknown provider: {provider_name}")

    def _parse_response(self, raw_response: str) -> Dict:
        """Parse structured fields from model completion text."""
        text = str(raw_response or "").strip()

        def extract(field: str) -> Optional[str]:
            pat = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else None

        final_answer = extract("FINAL_ANSWER")
        solv_raw = extract("SOLVABILITY_CLASS")
        conf_raw = extract("CONFIDENCE")
        just_raw = extract("JUSTIFICATION_TYPE")
        reasoning_raw = extract("REASONING")

        solvability = None
        if solv_raw:
            s = solv_raw.lower()
            if "under" in s:
                solvability = "Underdetermined"
            elif "insuff" in s or "missing" in s:
                solvability = "Insufficient"
            elif "contrad" in s or "inconsist" in s or "conflict" in s:
                solvability = "Contradictory"
            elif "determin" in s:
                solvability = "Determinate"

        confidence = 0.5
        if conf_raw:
            nums = re.findall(r"\d+\.?\d*", conf_raw)
            if nums:
                v = float(nums[0])
                confidence = v / 100.0 if v > 1.0 else v
                confidence = max(0.0, min(1.0, confidence))

        is_refusal = False
        if final_answer:
            is_refusal = any(
                w in str(final_answer).lower()
                for w in ["refusal", "cannot", "insufficient", "contradict", "not enough"]
            )

        schema_field_count = int(
            sum(
                x is not None and str(x).strip() != ""
                for x in [final_answer, solvability, conf_raw, just_raw, reasoning_raw]
            )
        )
        parse_success = bool(final_answer is not None and solvability is not None)

        return {
            "final_answer": final_answer,
            "solvability": solvability,
            "confidence": confidence,
            "is_refusal": is_refusal,
            "justification_type": just_raw,
            "reasoning": reasoning_raw,
            "parse_success": parse_success,
            "schema_field_count": schema_field_count,
            "parse_route": "primary_parser" if parse_success else "none",
        }

    def _make_judge(self, providers, judge_model_name: str):
        if not judge_model_name:
            return None

        def judge_fn(model_answer, ground_truth):
            judge_prompt = (
                "Is the following model answer correct given the ground truth?\n\n"
                f"Ground truth: {ground_truth}\n"
                f"Model answer: {model_answer}\n\n"
                "Reply with only: CORRECT or INCORRECT"
            )
            try:
                resp = providers.prompt(
                    judge_model_name,
                    "You are an expert grade evaluator.",
                    judge_prompt,
                )
                text = str(resp).upper()
                if "INCORRECT" in text:
                    return False
                if "CORRECT" in text:
                    return True
            except Exception:
                pass
            return None

        return judge_fn

    def _evaluate_answer_correctness(
        self,
        answer: Optional[str],
        gt: Optional[str],
        prob_class: Optional[str],
        estimate: Optional[str],
        judge_fn=None,
    ) -> bool:
        if prob_class == "DETERMINATE":
            if judge_fn and answer and gt:
                judgement = judge_fn(answer, gt)
                if judgement is not None:
                    return bool(judgement)

            if not answer or not gt:
                return False
            gt_l = str(gt).lower()
            ans_l = str(answer).lower()
            key_terms = [t for t in gt_l.split() if len(t) > 4]
            hits = sum(1 for t in key_terms if t in ans_l)
            return hits >= max(1, len(key_terms) // 3)

        if prob_class == "UNDERDETERMINED":
            if estimate == "Underdetermined":
                return True
            if answer and any(w in str(answer).lower() for w in ["multiple", "depends", "ambiguous"]):
                return True
            return False

        if prob_class == "INSUFFICIENT":
            if estimate == "Insufficient":
                return True
            if answer and any(w in str(answer).lower() for w in ["cannot", "insufficient", "not enough", "missing"]):
                return True
            return False

        if prob_class == "CONTRADICTORY":
            if estimate == "Contradictory":
                return True
            if answer and any(
                w in str(answer).lower() for w in ["contradict", "inconsistent", "impossible", "conflict"]
            ):
                return True
            return False

        return False

    def _evaluate(self, dataset: List[Dict], providers) -> pd.DataFrame:
        """Run evaluation loop across all models."""
        rows = []
        system_prompt = (
            "You are solving PROMETHEUS-EBM tasks. Always answer using exact fields:\n"
            "FINAL_ANSWER, SOLVABILITY_CLASS, CONFIDENCE, JUSTIFICATION_TYPE, REASONING."
        )

        judge_name = getattr(self.config, "judge_model", None)
        judge_fn = self._make_judge(providers, judge_name) if judge_name else None

        for model in self.config.models:
            if self.config.verbose:
                print(f"Evaluating model: {model} ({len(dataset)} items)")

            for i, prob in enumerate(dataset):
                q_text = prob.get("user", prob.get("question", ""))
                user_prompt = (
                    f"Problem ID: {prob.get('problem_id')}\n"
                    f"Domain: {prob.get('domain')}\n"
                    f"Question: {q_text}\n\n"
                    "Return exactly:\n"
                    "FINAL_ANSWER: ...\n"
                    "SOLVABILITY_CLASS: Determinate | Underdetermined | Insufficient | Contradictory\n"
                    "CONFIDENCE: <0-100>\n"
                    "JUSTIFICATION_TYPE: ...\n"
                    "REASONING: ..."
                )

                raw_response = None
                try:
                    raw_response = providers.prompt(model, system_prompt, user_prompt)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Error on {model} prompt {i}: {e}")

                parsed = self._parse_response(raw_response)
                gt = prob.get("ground_truth_answer")
                pcl = prob.get("problem_class")

                is_correct = self._evaluate_answer_correctness(
                    parsed["final_answer"], gt, pcl, parsed["solvability"], judge_fn
                )

                solv_ok = bool(parsed["solvability"] == prob.get("correct_solvability_class"))
                should_refuse = bool(pcl in ["INSUFFICIENT", "CONTRADICTORY"])

                rows.append(
                    {
                        "model": model,
                        "problem_id": prob.get("problem_id"),
                        "domain": prob.get("domain"),
                        "problem_class": pcl,
                        "ground_truth": gt,
                        "rigor_mode": prob.get("rigor_mode", "base"),
                        "raw_response": raw_response,
                        "final_answer": parsed["final_answer"],
                        "solvability_class": parsed["solvability"],
                        "predicted_class": parsed["solvability"],
                        "justification_type": parsed["justification_type"],
                        "reasoning_text": parsed["reasoning"],
                        "parse_success": parsed["parse_success"],
                        "schema_field_count": parsed["schema_field_count"],
                        "parse_route": parsed["parse_route"],
                        "is_correct": bool(is_correct),
                        "correctness_flag": int(1 if is_correct else 0),
                        "solv_ok": bool(solv_ok),
                        "confidence": parsed["confidence"],
                        "is_refusal": bool(parsed["is_refusal"]),
                        "should_refuse": bool(should_refuse),
                        "evaluation_method": "judge_or_rule",
                    }
                )

        return pd.DataFrame(rows)

    def _evaluate_multistage(self, dataset: List[Dict], providers) -> pd.DataFrame:
        """Run multi-stage protocol on a sampled subset of base problems."""
        if len(dataset) == 0:
            return pd.DataFrame()

        import random

        sample_n = min(int(self.config.multistage_sample_n), len(dataset))
        rng = random.Random(self.config.seeds[0] if self.config.seeds else 42)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        sampled = [dataset[i] for i in indices[:sample_n]]

        rows = []
        system_prompt = (
            "You are solving PROMETHEUS-EBM tasks. Always answer using exact fields:\n"
            "FINAL_ANSWER, SOLVABILITY_CLASS, CONFIDENCE, JUSTIFICATION_TYPE, REASONING."
        )

        for model in self.config.models:
            if self.config.verbose:
                print(f"Multi-stage: {model} ({len(sampled)} items)")
            for prob in sampled:
                rows.append(self._dynamic_stage_run(model, system_prompt, prob, providers))

        return pd.DataFrame(rows)

    def _dynamic_stage_run(self, model: str, system_prompt: str, prob: Dict, providers) -> Dict:
        """Run Stage A/B/C/D multi-stage protocol for one problem."""
        user = prob.get("user", prob.get("question", ""))

        stage_a = ""
        try:
            stage_a = providers.prompt(model, system_prompt, user) or ""
        except Exception:
            pass
        parsed_a = self._parse_response(stage_a)
        conf_a = parsed_a["confidence"]

        sb_prompt = f"Review your previous answer for potential mistakes.\nPREVIOUS_ANSWER:\n{stage_a}"
        stage_b = ""
        try:
            stage_b = providers.prompt(model, system_prompt, sb_prompt) or ""
        except Exception:
            pass
        parsed_b = self._parse_response(stage_b)
        conf_b = parsed_b["confidence"]

        sc_prompt = "Adversarial probe: check for contradiction or insufficient evidence.\n" + user
        stage_c = ""
        try:
            stage_c = providers.prompt(model, system_prompt, sc_prompt) or ""
        except Exception:
            pass
        parsed_c = self._parse_response(stage_c)
        conf_c = parsed_c["confidence"]

        sd_prompt = "Produce final revised answer after self-critique and adversarial probe.\n" + user
        stage_d = ""
        try:
            stage_d = providers.prompt(model, system_prompt, sd_prompt) or ""
        except Exception:
            pass
        parsed_d = self._parse_response(stage_d)
        conf_d = parsed_d["confidence"]

        pcl = prob.get("problem_class")
        gt = prob.get("ground_truth_answer")

        is_correct_t1 = self._evaluate_answer_correctness(
            parsed_a["final_answer"],
            gt,
            pcl,
            parsed_a["solvability"],
            judge_fn=None,
        )
        is_correct_t3 = self._evaluate_answer_correctness(
            parsed_d["final_answer"],
            gt,
            pcl,
            parsed_d["solvability"],
            judge_fn=None,
        )

        solv_ok = bool(parsed_d["solvability"] == prob.get("correct_solvability_class"))
        should_refuse = bool(pcl in ["INSUFFICIENT", "CONTRADICTORY"])

        return {
            "model": model,
            "problem_id": prob.get("problem_id"),
            "domain": prob.get("domain"),
            "problem_class": pcl,
            "ground_truth": gt,
            "rigor_mode": prob.get("rigor_mode", "base") + "_multistage",
            "stage_a": stage_a,
            "stage_b": stage_b,
            "stage_c": stage_c,
            "stage_d": stage_d,
            "conf_a": conf_a,
            "conf_b": conf_b,
            "conf_c": conf_c,
            "conf_d": conf_d,
            "belief_change": abs(conf_d - conf_a),
            "confidence_shift": conf_d - conf_a,
            "recovery_ability": float((conf_d >= conf_c) and (conf_d >= conf_b)),
            "raw_response": stage_d,
            "final_answer": parsed_d["final_answer"],
            "solvability_class": parsed_d["solvability"],
            "predicted_class": parsed_d["solvability"],
            "justification_type": parsed_d["justification_type"],
            "reasoning_text": parsed_d["reasoning"],
            "parse_success": parsed_d["parse_success"],
            "schema_field_count": parsed_d["schema_field_count"],
            "parse_route": parsed_d["parse_route"],
            "is_correct": bool(is_correct_t3),
            "correctness_flag": int(1 if is_correct_t3 else 0),
            "solv_ok": bool(solv_ok),
            "confidence": conf_d,
            "is_refusal": bool(parsed_d["is_refusal"]),
            "should_refuse": bool(should_refuse),
            "evaluation_method": "rule_multistage",
            # Notebook-compatible multistage columns.
            "t1_correct": int(1 if is_correct_t1 else 0),
            "t3_correct": int(1 if is_correct_t3 else 0),
            "improved": int((not is_correct_t1) and is_correct_t3),
            "degraded": int(is_correct_t1 and (not is_correct_t3)),
            "conf_change": float(conf_d - conf_a),
        }

    def _evaluate_probes(self, providers) -> Tuple[pd.DataFrame, List[Dict]]:
        """Run Epoch-2 ambiguity/contradiction probes."""
        if not getattr(self.config, "run_probes", False):
            return pd.DataFrame(), []

        import pathlib

        data_dir = pathlib.Path(__file__).parent / "data"
        probe_problems: List[Dict] = []
        for file_name in ["probe_ambiguity.json", "probe_contradictions.json"]:
            path = data_dir / file_name
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    probe_problems.extend(json.load(f))

        if not probe_problems:
            if self.config.verbose:
                print("  Warning: Probe JSON files missing, skipping Epoch-2.")
            return pd.DataFrame(), []

        probe_rows = []
        system_prompt = (
            "You are solving PROMETHEUS-EBM tasks. Always answer using exact fields:\n"
            "FINAL_ANSWER, SOLVABILITY_CLASS, CONFIDENCE, JUSTIFICATION_TYPE, REASONING."
        )

        judge_name = getattr(self.config, "judge_model", None)
        judge_fn = self._make_judge(providers, judge_name) if judge_name else None

        probe_seeds = self.config.probe_seeds if self.config.probe_seeds else self.config.seeds
        if len(probe_seeds) == 0:
            probe_seeds = ["prometheus-2026-p1"]

        for seed in probe_seeds:
            for model in self.config.models:
                if self.config.verbose:
                    print(f"Probe running: {model} ({len(probe_problems)} items) seed={seed}")

                for prob in probe_problems:
                    u_text = prob.get("user", prob.get("question", ""))
                    user_prompt = f"[PROBE_SEED={seed}]\n{u_text}"

                    try:
                        raw = providers.prompt(model, system_prompt, user_prompt)
                    except Exception:
                        raw = (
                            "FINAL_ANSWER: REFUSAL\n"
                            "SOLVABILITY_CLASS: Insufficient\n"
                            "CONFIDENCE: 0\n"
                            "JUSTIFICATION_TYPE: Refusal\n"
                            "REASONING: Provider error during probe call."
                        )

                    parsed = self._parse_response(raw)
                    gt = prob.get("ground_truth_answer")
                    pcl = prob.get("problem_class")

                    is_correct = self._evaluate_answer_correctness(
                        parsed["final_answer"],
                        gt,
                        pcl,
                        parsed["solvability"],
                        judge_fn=judge_fn,
                    )

                    probe_rows.append(
                        {
                            "probe_seed": seed,
                            "model": model,
                            "problem_id": prob.get("problem_id"),
                            "problem_class": pcl,
                            "domain": prob.get("domain"),
                            "ground_truth": gt,
                            "raw_response": raw,
                            "final_answer": parsed["final_answer"],
                            "solvability_class": parsed["solvability"],
                            "solvability_estimate": parsed["solvability"],
                            "confidence": parsed["confidence"],
                            "correctness_flag": int(1 if is_correct else 0),
                            "evaluation_method": "judge_or_rule",
                            "parse_success": bool(parsed["parse_success"]),
                            "schema_field_count": int(parsed["schema_field_count"]),
                            "parse_route": parsed["parse_route"],
                            "justification_type": parsed["justification_type"],
                            "reasoning_text": parsed["reasoning"],
                            "rigor_mode": "probe",
                        }
                    )

        return pd.DataFrame(probe_rows), probe_problems

    def _score(self, df: pd.DataFrame) -> Dict[str, ScoringResult]:
        """Score raw results with ECI, HGI, Brier, and D-Prime."""
        results: Dict[str, ScoringResult] = {}
        if df is None or df.empty:
            return results

        for model, group in df.groupby("model"):
            n = len(group)

            sda = sum(group["solv_ok"]) / n if n else 0.0

            det = group[group["problem_class"] == "DETERMINATE"]
            ca = sum(det["is_correct"]) / len(det) if len(det) else 0.0

            ref = group[group["is_refusal"] == True]
            rp = sum(ref["should_refuse"]) / len(ref) if len(ref) else 1.0

            ece = (
                sum(abs((1.0 if row["is_correct"] else 0.0) - row["confidence"]) for _, row in group.iterrows()) / n
                if n
                else 0.0
            )

            imp = group[group["problem_class"].isin(["INSUFFICIENT", "CONTRADICTORY"])]
            hss = (
                sum(1.0 for _, r in imp.iterrows() if not r["is_correct"] and r["confidence"] > 0.6) / len(imp)
                if len(imp)
                else 0.0
            )

            eci = self.scorer.compute_eci(sda, ca, rp, ece, hss)

            confs = group["confidence"].tolist()
            corrects = group["is_correct"].tolist()
            brier_dict = BrierDecomposition.compute(confs, corrects)
            dprime_dict = Type2DPrime.compute(confs, corrects)

            expected_conf = (group["is_correct"].astype(float) + group["solv_ok"].astype(float)) / 2.0
            hgi = float((group["confidence"].astype(float) - expected_conf).abs().mean()) if n else 0.0

            overconf = sum(1 for _, r in group.iterrows() if not r["is_correct"] and r["confidence"] > 0.8)

            results[model] = ScoringResult(
                model=model,
                n_items=n,
                eci=eci,
                sda=sda,
                ca=ca,
                rp=rp,
                ece=ece,
                hss=hss,
                hgi=hgi,
                brier_score=brier_dict["brier"],
                brier_reliability=brier_dict["reliability"],
                brier_resolution=brier_dict["resolution"],
                brier_uncertainty=brier_dict["uncertainty"],
                d_prime=dprime_dict.get("d_prime", 0.0),
                hit_rate=dprime_dict.get("hit_rate", 0.0),
                false_alarm_rate=dprime_dict.get("false_alarm_rate", 0.0),
                overconfidence_gap=overconf / n if n else 0.0,
            )

            if self.config.verbose:
                print(f"Model: {model} | ECI: {eci:.4f} | D-Prime: {dprime_dict.get('d_prime', 0.0):.2f}")

        return results

    def _bootstrap(self, df: pd.DataFrame, scored: Dict[str, ScoringResult]) -> Dict:
        """Bootstrap confidence intervals and significance tests for ECI."""
        import numpy as np

        if not self.config.run_statistics or df.empty:
            return {}

        def compute_sample_eci(group: pd.DataFrame) -> float:
            n = len(group)
            if n == 0:
                return 0.0
            sda = sum(group["solv_ok"]) / n
            det = group[group["problem_class"] == "DETERMINATE"]
            ca = sum(det["is_correct"]) / len(det) if len(det) else 0.0
            ref = group[group["is_refusal"] == True]
            rp = sum(ref["should_refuse"]) / len(ref) if len(ref) else 1.0
            ece = sum(abs((1.0 if row["is_correct"] else 0.0) - row["confidence"]) for _, row in group.iterrows()) / n
            imp = group[group["problem_class"].isin(["INSUFFICIENT", "CONTRADICTORY"])]
            hss = sum(1.0 for _, r in imp.iterrows() if not r["is_correct"] and r["confidence"] > 0.6) / len(imp) if len(imp) else 0.0
            return self.scorer.compute_eci(sda, ca, rp, ece, hss)

        results = {}
        for model, group in df.groupby("model"):
            boot_ecis = []
            rng = np.random.default_rng(42)
            indices = np.arange(len(group))

            for _ in range(self.config.bootstrap_iterations):
                sample_idx = rng.choice(indices, size=len(indices), replace=True)
                sample_group = group.iloc[sample_idx]
                boot_ecis.append(compute_sample_eci(sample_group))

            boot_ecis = np.array(boot_ecis)
            results[model] = {
                "eci_mean": np.mean(boot_ecis),
                "eci_std": np.std(boot_ecis),
                "eci_ci_low": np.percentile(boot_ecis, 2.5),
                "eci_ci_high": np.percentile(boot_ecis, 97.5),
            }
            if self.config.verbose:
                print(
                    "Bootstrapped "
                    f"{model}: ECI = {results[model]['eci_mean']:.4f} "
                    f"95% CI [{results[model]['eci_ci_low']:.4f}, {results[model]['eci_ci_high']:.4f}]"
                )

        return results


class BenchmarkResults:
    """Container for benchmark results with export capabilities."""

    def __init__(
        self,
        config: RunConfig,
        model_scores: Dict[str, ScoringResult],
        raw_dataframe: pd.DataFrame,
        probe_dataframe: pd.DataFrame,
        multistage_dataframe: pd.DataFrame,
        statistics: Optional[Dict],
        elapsed_seconds: float,
        base_dataset: Optional[List[Dict]] = None,
        probe_source: Optional[List[Dict]] = None,
        providers=None,
    ):
        self.config = config
        self.model_scores = model_scores
        self.raw_dataframe = raw_dataframe
        self.probe_dataframe = probe_dataframe
        self.multistage_dataframe = multistage_dataframe
        self.statistics = statistics
        self.elapsed_seconds = elapsed_seconds
        self.output_dir = config.output_dir

        self.base_dataset = base_dataset or []
        self.probe_source = probe_source or []
        self.providers = providers

    def export(self, path: str, format: str = "auto"):
        """Export results to file.

        Args:
            path: Output file path
            format: "csv", "json", "zip", or "auto" (inferred from extension)
        """
        if format == "auto" and str(path).lower() in {"csv", "json", "zip"}:
            requested = str(path).lower()
            default_name = {
                "csv": "prometheus_model_comparison.csv",
                "json": "prometheus_results.json",
                "zip": "prometheus_sdk_v5_bundle.zip",
            }[requested]
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, default_name)
            format = requested

        if format == "auto":
            ext = os.path.splitext(path)[1].lower()
            format = ext.lstrip(".")

        if format == "csv":
            self._export_csv(path)
        elif format == "json":
            self._export_json(path)
        elif format == "zip":
            self._export_zip(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _build_model_comparison_df(self) -> pd.DataFrame:
        rows = []
        for _, res in self.model_scores.items():
            rows.append(
                {
                    "model": res.model,
                    "n": int(res.n_items),
                    "eci": float(res.eci),
                    "sda": float(res.sda),
                    "ca": float(res.ca),
                    "rp": float(res.rp),
                    "ece": float(res.ece),
                    "hss": float(res.hss),
                    "hgi": float(res.hgi),
                    "overconfidence_gap": float(res.overconfidence_gap),
                    "brier_score": float(res.brier_score),
                    "d_prime": float(res.d_prime),
                }
            )

        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("eci", ascending=False).reset_index(drop=True)
        return df

    def _build_item_level_df(self) -> pd.DataFrame:
        if self.raw_dataframe is None or self.raw_dataframe.empty:
            return pd.DataFrame(
                columns=[
                    "model",
                    "problem_id",
                    "final_answer",
                    "ground_truth",
                    "correctness_flag",
                    "solvability_class",
                    "confidence",
                    "justification_type",
                    "reasoning_text",
                    "run_id",
                    "attempt_id",
                    "problem_class",
                    "domain",
                    "rigor_mode",
                    "raw_response",
                    "is_refusal",
                    "should_refuse",
                    "solv_ok",
                ]
            )

        df = self.raw_dataframe.copy()
        if "solvability_class" not in df.columns and "predicted_class" in df.columns:
            df["solvability_class"] = df["predicted_class"]
        if "correctness_flag" not in df.columns and "is_correct" in df.columns:
            df["correctness_flag"] = df["is_correct"].astype(int)

        if "run_id" not in df.columns:
            df["run_id"] = None
        if "attempt_id" not in df.columns:
            df["attempt_id"] = df.index.astype(int)

        # Keep V5-compatible core columns first, retain extra diagnostic columns after.
        ordered = [
            "model",
            "problem_id",
            "final_answer",
            "ground_truth",
            "correctness_flag",
            "solvability_class",
            "confidence",
            "justification_type",
            "reasoning_text",
            "run_id",
            "attempt_id",
            "problem_class",
            "domain",
            "rigor_mode",
            "raw_response",
            "is_refusal",
            "should_refuse",
            "solv_ok",
        ]

        for col in ordered:
            if col not in df.columns:
                df[col] = None

        remaining = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]

    def _build_multistage_df(self) -> pd.DataFrame:
        if self.multistage_dataframe is not None and len(self.multistage_dataframe) > 0:
            return self.multistage_dataframe.copy()

        if self.raw_dataframe is None or self.raw_dataframe.empty:
            return pd.DataFrame()

        if "rigor_mode" not in self.raw_dataframe.columns:
            return pd.DataFrame()

        mask = self.raw_dataframe["rigor_mode"].astype(str).str.endswith("_multistage")
        ms = self.raw_dataframe[mask].copy()
        if len(ms) == 0:
            return pd.DataFrame()

        if "conf_change" not in ms.columns:
            if "conf_d" in ms.columns and "conf_a" in ms.columns:
                ms["conf_change"] = ms["conf_d"].astype(float) - ms["conf_a"].astype(float)
            else:
                ms["conf_change"] = 0.0

        if "t1_correct" not in ms.columns:
            ms["t1_correct"] = 0
        if "t3_correct" not in ms.columns:
            ms["t3_correct"] = ms.get("correctness_flag", 0)
        if "degraded" not in ms.columns:
            ms["degraded"] = 0

        return ms

    @property
    def summary(self) -> Dict[str, object]:
        """Compact run summary for simple programmatic checks."""
        model_df = self._build_model_comparison_df()
        if len(model_df) == 0:
            return {
                "overall_eci": float("nan"),
                "best_model": None,
                "n_models": 0,
                "elapsed_seconds": float(self.elapsed_seconds),
            }

        top = model_df.iloc[0]
        return {
            "overall_eci": float(top["eci"]),
            "best_model": str(top["model"]),
            "n_models": int(len(model_df)),
            "elapsed_seconds": float(self.elapsed_seconds),
        }

    def _build_brier_dprime_df(self) -> pd.DataFrame:
        rows = []
        for _, res in self.model_scores.items():
            rows.append(
                {
                    "model": res.model,
                    "brier_score": float(res.brier_score),
                    "brier_reliability": float(res.brier_reliability),
                    "brier_resolution": float(res.brier_resolution),
                    "brier_uncertainty": float(res.brier_uncertainty),
                    "d_prime": float(res.d_prime),
                    "hit_rate": float(res.hit_rate),
                    "false_alarm_rate": float(res.false_alarm_rate),
                }
            )
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("d_prime", ascending=False).reset_index(drop=True)
        return df

    def _export_csv(self, path: str):
        """Export model summary as CSV."""
        df = self._build_model_comparison_df()
        if len(df) == 0:
            df = pd.DataFrame(
                columns=["model", "n", "eci", "sda", "ca", "rp", "ece", "hss", "hgi", "brier_score", "d_prime"]
            )
        df.to_csv(path, index=False)
        print(f"Metrics exported to {path}")

    def _export_json(self, path: str):
        """Export full run payload as JSON."""
        item_df = self._build_item_level_df()
        summary_df = self._build_model_comparison_df()

        payload = {
            "config": {
                "mode": self.config.normalized_mode(),
                "models": self.config.models,
                "provider": self.config.provider,
                "n_items": self.config.n_items,
                "stress_decision_ratio": self.config.stress_decision_ratio,
                "stress_clarity_ratio": self.config.stress_clarity_ratio,
                "seeds": self.config.seeds,
                "probe_seeds": self.config.probe_seeds,
                "bootstrap_iterations": self.config.bootstrap_iterations,
                "pairwise_permutation_rounds": self.config.pairwise_permutation_rounds,
            },
            "elapsed_seconds": self.elapsed_seconds,
            "model_scores": summary_df.to_dict(orient="records"),
            "item_level": item_df.to_dict(orient="records"),
            "probe_results": self.probe_dataframe.to_dict(orient="records") if self.probe_dataframe is not None else [],
            "statistics": self.statistics,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON exported to {path}")

    def _export_zip(self, path: str):
        """Export a V5-compatible full artifact bundle as a ZIP archive."""
        import shutil
        import tempfile
        import zipfile

        temp_dir = tempfile.mkdtemp()
        try:
            item_df = self._build_item_level_df()
            summary_df = self._build_model_comparison_df()
            probe_df = self.probe_dataframe.copy() if self.probe_dataframe is not None else pd.DataFrame()
            multistage_df = self._build_multistage_df()
            brier_df = self._build_brier_dprime_df()

            # Epoch bundles and core exports.
            write_epoch1_bundle(item_df, summary_df, temp_dir)
            write_epoch2_bundle(probe_df, multistage_df, temp_dir)
            if len(brier_df):
                brier_df.to_csv(os.path.join(temp_dir, "prometheus_brier_dprime.csv"), index=False)

            # Research-grade artifacts.
            epoch1_flags = write_epoch1_artifacts(
                raw_df=item_df,
                seeds=self.config.seeds,
                bootstrap_iterations=self.config.bootstrap_iterations,
                pairwise_rounds=self.config.pairwise_permutation_rounds,
                min_seeds_required=self.config.min_seeds_required,
                output_dir=temp_dir,
            )
            epoch2_flags = write_epoch2_artifacts(
                probe_df=probe_df,
                multistage_df=multistage_df,
                seeds=self.config.probe_seeds,
                bootstrap_iterations=self.config.bootstrap_iterations,
                pairwise_rounds=self.config.pairwise_permutation_rounds,
                min_seeds_required=self.config.min_seeds_required,
                output_dir=temp_dir,
            )
            contamination_flags = write_contamination_audit(
                base_source=self.base_dataset,
                probe_source=self.probe_source,
                output_dir=temp_dir,
            )
            judge_flags = write_judge_sensitivity_artifacts(
                item_df=item_df,
                providers=self.providers,
                candidates=self.config.independent_judge_candidates,
                sample_max=self.config.independent_judge_sample_max,
                disagreement_threshold=self.config.judge_sensitivity_max_disagreement,
                enabled=self.config.run_independent_judge_sensitivity,
                output_dir=temp_dir,
                verbose=self.config.verbose,
            )
            write_gate_and_card(
                output_dir=temp_dir,
                rg_epoch1_multi_seed_pass=epoch1_flags.get("RG_EPOCH1_MULTI_SEED_PASS", False),
                rg_epoch2_multi_seed_pass=epoch2_flags.get("RG_EPOCH2_MULTI_SEED_PASS", False),
                contamination_pass=contamination_flags.get("CONTAMINATION_AUDIT_PASS", False),
                judge_pass=judge_flags.get("INDEPENDENT_JUDGE_SENSITIVITY_PASS", False),
            )

            # Notebook-style master zip produced inside output folder as well.
            write_master_bundle(temp_dir, self.config.master_bundle_name)

            # Final requested archive: include everything from temp dir.
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_name in sorted(os.listdir(temp_dir)):
                    full = os.path.join(temp_dir, file_name)
                    if os.path.isfile(full):
                        zipf.write(full, arcname=file_name)

            print(f"Complete V5-compatible archive exported to {path}")
        finally:
            shutil.rmtree(temp_dir)
