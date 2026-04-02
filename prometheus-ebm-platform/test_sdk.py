"""Quick smoke test for the PROMETHEUS-EBM SDK."""
from prometheus_ebm import Dataset, ECIScorer, ModelProfiler
from prometheus_ebm.taxonomy import SolvabilityClass
from prometheus_ebm.stats import bootstrap_ci, cohens_h, effect_magnitude
from prometheus_ebm.stress import StressAugmenter

# 1. Load dataset
ds = Dataset.load_default()
stats = ds.stats()
print("PROMETHEUS-EBM SDK v0.1.0 - WORKING")
print(f"Dataset: {stats['total']} problems")
print(f"Classes: {stats['by_class']}")
print(f"Domains: {stats['by_domain']}")

# 2. Test filtering
financial = ds.filter(domain=ds.problems[0].domain)
print(f"\nFiltered (first domain): {len(financial)} problems")

by_class = ds.by_class()
print(f"By class: {', '.join(f'{k.name}={len(v)}' for k, v in by_class.items())}")

# 3. Test stress augmentation
augmenter = StressAugmenter()
augmented = augmenter.augment(ds.problems)
print(f"\nAugmented: {len(ds)} -> {len(augmented)} problems (+{len(augmented)-len(ds)} stress variants)")

# 4. Test stats
vals = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
mean, lo, hi = bootstrap_ci(vals)
print(f"\nBootstrap CI test: mean={mean:.2f} [{lo:.2f}, {hi:.2f}]")

h = cohens_h(0.80, 0.65)
print(f"Cohen's h test: h={h:.3f} ({effect_magnitude(h)})")

# 5. Test solvability parsing
s = SolvabilityClass.from_string("underdetermined")
print(f"\nParsing test: 'underdetermined' -> {s.name}")
print(f"  is_solvable: {s.is_solvable}")
print(f"  is_impossible: {s.is_impossible}")
print(f"  is_ambiguous: {s.is_ambiguous}")

print("\n" + "="*50)
print("ALL TESTS PASSED")
print("="*50)
