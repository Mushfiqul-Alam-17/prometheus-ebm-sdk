# Kaggle Run Guide for benchmark-v4

## What to upload to Kaggle Input

Upload these 3 files as a Kaggle dataset (already prepared in `kaggle/input_bundle/`):

1. `prometheus_real_dataset.json`
2. `probe_ambiguity.json`
3. `probe_contradictions.json`

## Do I need to upload the SDK?

Not required for `benchmark-v4.ipynb`.

- The notebook has fallback implementations for core scoring/generation logic.
- If optional external modules (`problem_generator.py`, `scoring_engine.py`) are not present, the notebook still runs using fallback code.

Optional:
- You can still upload SDK files for consistency, but it is not required for a successful benchmark-v4 run.

## Is CI added?

Yes for metrics confidence intervals and significance testing inside the notebook:

- RG01 utilities implement bootstrap CIs and permutation tests.
- RG02 and RG05 generate multi-seed CI/significance artifacts.

If you mean software CI/CD pipelines (e.g., GitHub Actions), those are not part of this local folder setup.

## Kaggle Execution Order

Run in this order:

1. C03-C17 (Epoch-1 and export)
2. P01-P05 (Epoch-2 probes, multistage, strict epoch2 export)
3. RG01-RG06 (research-grade block)

## Required Final Outputs to Verify

- `research_grade_v1_gate.json`
- `research_grade_v1_gate_criteria.csv`
- `benchmark_card_research_grade_v1.md`
- `rg_epoch1_eci_hgi_ci.csv`
- `rg_epoch1_pairwise_significance.csv`
- `contamination_audit_report.json`
- `independent_judge_sensitivity_report.json`
- `rg_epoch2_ci_summary.csv`
- `rg_epoch2_pairwise_significance.csv`

## Post-Run Research v1 Analysis (No New Data)

After downloading outputs locally into `data/results/results/`, run:

```powershell
python tools/research_v1/hypothesis_analysis.py
```

This generates:

- `outputs/research_grade/research_v1_model_epistemic_profile.csv`
- `outputs/research_grade/research_v1_rts_item_level.csv`
- `outputs/research_grade/research_v1_hypothesis_report.json`

These files quantify:

- Epistemic Dunning-Kruger trend (model quality vs overconfident wrong-rate)
- Reasoning Transparency Score (RTS) signal vs correctness and solvability detection

## Post-Run Dataset Decision Record

To avoid forgetting what to review after full v4 results, use:

- `docs/POSTRUN_DATASET_DECISION_PLAN.md`

This file contains:

- frozen pre-run dataset baseline
- keep-vs-fix decision criteria
- fix-priority order (if needed)
- versioning and change-control rules
- fill-in template for the final decision

## Common Pitfalls

- Running RG cells before base/probe cells are completed.
- Missing Kaggle input files (dataset/probes).
- Assuming old local `data/results/` artifacts are from v4; always trust current run outputs only.
