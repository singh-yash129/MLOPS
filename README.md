# IRIS MLOps Pipeline — MLflow Integration (Week 5)

Experiment tracking and a model registry for the IRIS classification pipeline using **MLflow**. Every training run logs hyperparameters, evaluation metrics, and the trained model, so experiments can be compared side-by-side and the best model served directly from a central registry — replacing DVC-based model storage.

## Pipeline Overview

```
Training Loop
      │
      ▼
MLflow
      │
      ├── Experiment Tracking   (params + metrics + artifacts per run)
      └── Model Registry        (version + fetch by name/version)
      │
      ▼
Evaluation / Inference
```

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── cp.yml                     # CI: DVC pull (data), MLflow train, registry-based pytest
├── .dvc/
│   └── config                         # DVC remote (GCS bucket) — data only, models removed in Week 5
├── dvc_data/
│   ├── iris_iter_1.csv.dvc
│   ├── iris_iter_2.csv.dvc
│   └── iris_iter_3.csv.dvc
├── scripts/
│   ├── train.py                       # Week 2/4 training script (DVC-versioned model, now unused for models)
│   ├── train_mlflow.py                # Hyperparameter tuning + MLflow experiment tracking
│   └── evaluate_from_registry.py      # Loads model from MLflow Model Registry
├── tests/
│   ├── test_data_validation.py        # Schema, nulls, types, value-range checks
│   └── test_model_registry.py         # Model fetched from MLflow Registry, metric thresholds
├── mlflow.db                          # MLflow SQLite tracking store (generated, gitignored)
├── requirements.txt
└── README.md
```

## Prerequisites

- GCP project with a GCS bucket configured as the DVC remote (from Week 2) — used for **data only** from this week onward
- A GCP service account with `roles/storage.objectViewer` on that bucket
- Workload Identity Federation configured between the service account and this GitHub repository (see [Authentication](#authentication))
- Python 3.12 (required by `scikit-learn==1.9.0`)

## Authentication

Since GCP service account **key creation is disabled by org policy**, this pipeline authenticates using **Workload Identity Federation (WIF)** — no JSON key is ever downloaded or stored.

Two GitHub repository secrets are required (**Settings → Secrets and variables → Actions**):

| Secret | Value |
|---|---|
| `WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `WIF_SERVICE_ACCOUNT` | `github-ci-dvc@<PROJECT_ID>.iam.gserviceaccount.com` |

## MLflow Tracking Backend

MLflow's plain filesystem store (`file:./mlruns`) is in maintenance mode in the version used here. This pipeline uses the **SQLite backend** instead:

```
sqlite:///mlflow.db
```

`mlflow.db`, `mlruns/`, and `mlartifacts/` are git-ignored and regenerated fresh on each local run or CI run — they are not committed to the repository.

## Workflow (`.github/workflows/cp.yml`)

Triggers on every push and pull request, across **all branches**:

```yaml
on:
  push:
    branches: ['**']
  pull_request:
    branches: ['**']
```

Steps:
1. Checkout the repository, install dependencies (Python 3.12)
2. Authenticate to GCP via WIF
3. `dvc pull` — fetch versioned **data** (models are no longer DVC-tracked)
4. Run `scripts/train_mlflow.py` — hyperparameter tuning across multiple configurations, logging each run's parameters, metrics, and model to MLflow, registering the model under `iris_random_forest`
5. Run the `pytest` suite — `test_model_registry.py` fetches the just-registered model directly from the MLflow Registry and validates accuracy, precision, recall, and F1 against minimum thresholds
6. Upload the pytest report and the MLflow tracking database as workflow artifacts

## Tasks Covered

| Task | Description |
|---|---|
| 1 | Hyperparameter tuning — 3 configurations varying `n_estimators` and `max_depth` |
| 2 | MLflow logging — parameters, metrics, and model artifact for every run |
| 3 | Compare experiments in the MLflow Tracking UI |
| 4 | Model artifacts removed from DVC; DVC now tracks data only |
| 5 | Evaluation pipeline fetches the model by name/version from the MLflow Model Registry |
| 6 (optional) | CI trains and fetches from the MLflow Registry in the same workflow run |

## Running Tests Locally

```bash
pip install -r requirements.txt
dvc pull

# Task 1 & 2
python3 scripts/train_mlflow.py --iteration 3

# Task 3
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000 -> iris_classification experiment -> select runs -> Compare

# Task 5
python3 scripts/evaluate_from_registry.py
pytest tests/test_model_registry.py -v
```

## Notes

- Metric thresholds in `tests/test_model_registry.py` (`MIN_ACCURACY`, `MIN_PRECISION`, `MIN_RECALL`, `MIN_F1`) should be tuned to match expected model performance.
- `scripts/train.py` and any Week 4 DVC-based model tests are retained for history only — models are no longer read from DVC-tracked paths as of this week.
- Data validation in Week 4 previously caught a real synthetic-augmentation bug (near-zero `petal width` from unclipped noise), fixed at the source in `scripts/train.py`'s clipping logic — the same clipping is reused in `train_mlflow.py`.