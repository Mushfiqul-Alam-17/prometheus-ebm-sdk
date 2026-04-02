# Post-Run Dataset Decision Plan (v4)

Purpose: preserve what to review after the full v4 run so dataset changes are decided by evidence, not memory.

Date saved: 2026-03-28
Scope: research-grade v1 only (no human-feedback or paper-writing tasks)

## Current Pre-Run Baseline (Locked)

Use current Kaggle input bundle as the frozen baseline for v4 first full run:

- kaggle/input_bundle/prometheus_200_multimodel_dataset.json
- kaggle/input_bundle/probe_ambiguity.json
- kaggle/input_bundle/probe_contradictions.json

### Pre-Run Audit Snapshot

- Base count: 200
- Probe counts: ambiguity 25, contradiction 20 (total 45)
- Base class balance: 50/50/50/50
- Base domain balance: 40 each across 5 domains
- Probe/base ID overlap: 0
- Missing required rows: 0 (base and probes)
- Determinate rows missing ground truth: 0
- Stable hash present/unique: 200/200

Conclusion before run: do not change dataset for v4 first full run.

## After Full v4 Run: Decision Workflow

Run these checks in order:

1. Confirm research-grade gate artifacts exist and load successfully:
   - research_grade_v1_gate.json
   - research_grade_v1_gate_criteria.csv
   - benchmark_card_research_grade_v1.md

2. Confirm all six gate criteria from research_grade_v1_gate.json.

3. Review class/domain slices and instability:
   - rg_epoch1_seed_summary.csv
   - rg_epoch1_eci_hgi_ci.csv
   - rg_epoch2_ci_summary.csv
   - rg_epoch2_pairwise_significance.csv

4. Review contamination/judge-sensitivity artifacts:
   - contamination_audit_report.json
   - independent_judge_sensitivity_report.json

5. Run hypothesis analysis (no new data):
   - python tools/research_v1/hypothesis_analysis.py
   - Inspect outputs in outputs/research_grade/

## Decision Matrix (Keep vs Fix)

### Keep dataset as-is (recommended) if all are true

- Gate claim is true and all 6 criteria pass.
- No contamination overlap violations.
- Judge sensitivity passes threshold.
- No severe class-specific collapse persists across seeds.
- Probe results are interpretable and consistent with expected task difficulty.

### Fix dataset (targeted v1.1) if any of these occur

- Repeated seed-instability in a specific class/domain that points to task ambiguity or label quality issues.
- Probe class/domain skew causes interpretation bias for key findings.
- Clear leakage, duplicate semantics, or contradictory labeling found in audit outputs.
- Determinate tasks with low discriminative value dominate metrics and mask metacognitive behavior.

## If Fix Is Needed, Fix in This Order

1. Quality fixes (highest priority)
   - Correct mislabeled rows, contradictory labels, malformed prompts.
   - Remove or rewrite ambiguous rows that are unintentionally underdetermined.

2. Probe balancing fixes
   - Balance probe domains per class where feasible.
   - Keep probe problem style consistent with benchmark schema.

3. Difficulty calibration fixes
   - Add harder determinate items only if determinate slice is saturated and non-discriminative.
   - Add explicit difficulty tags only after preserving comparability snapshot.

4. Reproducibility hardening
   - Version-lock dataset (v1.1) and produce checksum manifest.
   - Keep v4 baseline unchanged for historical comparability.

## Change-Control Rules

- Never overwrite current v4 baseline files directly.
- Create a versioned dataset package for any modification (for example: v4_1_ready).
- Record exactly what changed and why in a diff log.
- Re-run contamination and gate checks after any dataset modification.

## Post-Run Fill-In Template

- Full run date:
- Gate claim (true/false):
- Criteria failed (if any):
- Primary failure mode observed:
- Proposed dataset action: keep / patch / rebalance / relabel
- Scope of patch (rows/classes/domains):
- Version name for revised dataset (if needed):
- Owner and deadline:
