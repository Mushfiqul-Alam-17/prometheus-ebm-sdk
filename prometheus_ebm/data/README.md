# Dataset

The PROMETHEUS-EBM benchmark dataset consists of 200 curated problems across 5 professional domains and 4 solvability classes.

## Structure

Each problem in `prometheus_200_multimodel_dataset.json` contains:

```json
{
  "problem_id": "MED-DET-001",
  "domain": "medical",
  "problem_class": "DETERMINATE",
  "question": "...",
  "ground_truth_answer": "...",
  "ground_truth_solvability": "DETERMINATE",
  "difficulty": "medium"
}
```

## Distribution

| Domain | Determinate | Underdetermined | Insufficient | Contradictory | Total |
|--------|:-----------:|:---------------:|:------------:|:-------------:|:-----:|
| Medical | 10 | 10 | 10 | 10 | 40 |
| Financial | 10 | 10 | 10 | 10 | 40 |
| Legal | 10 | 10 | 10 | 10 | 40 |
| Environmental | 10 | 10 | 10 | 10 | 40 |
| Social | 10 | 10 | 10 | 10 | 40 |
| **Total** | **50** | **50** | **50** | **50** | **200** |

## Download

The dataset is bundled with the SDK when installed via pip. For manual download:

```bash
# Download from the Kaggle competition dataset
kaggle datasets download -d mushfiqulalam007/prometheus-ebm-dataset
```

Or copy from the Kaggle notebook output after running the benchmark.
