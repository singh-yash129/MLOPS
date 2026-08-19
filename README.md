# Week 9 — Explainability, Fairness & Drift in the IRIS Pipeline

Branch: `week_9`

## Setup

```bash
pip install -r requirements.txt
```

## Run all tasks (in order — each depends on outputs from Task 1)

```bash
python3 task1_train_with_location.py   # Task 1: adds location attribute, trains model
python3 task2_fairness_audit.py        # Task 2: Fairlearn fairness audit
python3 task3_shap_analysis.py         # Task 3: SHAP explainability plots
python3 task4_drift_detection.py       # Task 4: drift detection via KS test
```

Task 5 (`model_card.md`) is a static document — no script to run.

## Files

| File | Purpose |
|---|---|
| `task1_train_with_location.py` | Adds random `location` sensitive attribute, trains RandomForest on the 4 real features only |
| `task2_fairness_audit.py` | Fairlearn `MetricFrame` — accuracy/precision/recall disaggregated by `location` |
| `task3_shap_analysis.py` | SHAP `TreeExplainer` summary plots for all 3 classes, plain-language virginica breakdown |
| `task4_drift_detection.py` | Simulates production drift on petal features, KS test vs. training distribution |
| `model_card.md` | Task 5 — model card covering use, performance, limitations, fairness, monitoring |

## Generated output (created by running the scripts above)

```
data/
  iris_with_location.csv
  iris_test_with_location.csv
  fairness_metrics_by_group.csv
  iris_simulated_production.csv
  drift_test_results.csv
models/
  iris_model.joblib
plots/
  shap_summary_setosa.png
  shap_summary_versicolor.png
  shap_summary_virginica.png
  drift_sepal_length_cm.png
  drift_sepal_width_cm.png
  drift_petal_length_cm.png
  drift_petal_width_cm.png
```

## Actual results (from a verified run)

**Task 2 — Fairness by location group:**
| Group | Accuracy | Precision | Recall |
|---|---|---|---|
| 0 | 0.846 | 0.878 | 0.878 |
| 1 | 0.941 | 0.933 | 0.933 |

**Task 4 — Drift test (Kolmogorov–Smirnov):**
| Feature | p-value | Drifted? |
|---|---|---|
| sepal length | 1.000000 | No |
| sepal width | 1.000000 | No |
| petal length | 0.000000 | Yes |
| petal width | 0.000000 | Yes |

