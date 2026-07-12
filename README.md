# IRIS MLOps Pipeline — CI Integration

Continuous Integration for the IRIS classification pipeline using **GitHub Actions**, **DVC**, and **CML**. Every push and pull request automatically pulls versioned data/models, runs validation and evaluation tests, and posts results as a PR comment.

## Pipeline Overview

```
Git Push / PR
      │
      ▼
GitHub Actions
      │
      ├── dvc pull        (fetch data + model from GCS)
      ├── pytest           (data validation + model evaluation)
      └── CML              (post test report as PR comment)
      │
      ▼
Validated Pipeline
```

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI workflow: DVC pull, pytest, CML report
├── data/
│   ├── train.csv                   # DVC-tracked training data
│   └── eval.csv                    # DVC-tracked evaluation data
├── models/
│   └── model.pkl                   # DVC-tracked trained model
├── tests/
│   ├── test_data_validation.py     # Task 1: schema, nulls, types, value ranges
│   └── test_model_evaluation.py    # Task 2: accuracy, precision, recall, F1
├── requirements.txt
└── README.md
```

## Prerequisites

- GCP project with a GCS bucket configured as the DVC remote (from the previous week's assignment)
- A GCP service account with `roles/storage.objectViewer` on that bucket
- Workload Identity Federation configured between the service account and this GitHub repository (see [Authentication](#authentication))

## Authentication

Since GCP service account **key creation is disabled by org policy**, this pipeline authenticates using **Workload Identity Federation (WIF)** — no JSON key is ever downloaded or stored.

Two GitHub repository secrets are required (**Settings → Secrets and variables → Actions**):

| Secret | Value |
|---|---|
| `WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `WIF_SERVICE_ACCOUNT` | `github-ci-dvc@<PROJECT_ID>.iam.gserviceaccount.com` |

`GITHUB_TOKEN` (used by CML to post PR comments) is provided automatically by GitHub Actions — no setup needed.

## CI Workflow (`.github/workflows/ci.yml`)

Triggers on every push and pull request, across **all branches**:

```yaml
on:
  push:
    branches: ['**']
  pull_request:
    branches: ['**']
```

Jobs:
1. **test** — checks out the repo, installs dependencies, authenticates to GCP via WIF, runs `dvc pull`, then runs the full `pytest` suite. Uploads the pytest and metrics reports as workflow artifacts.
2. **cml-report** — re-runs the tests, then uses [CML](https://cml.dev) to post the metrics report as a comment on the pull request.

## Tests

### Data Validation (`tests/test_data_validation.py`)
- Expected schema / column presence
- No missing values
- Correct feature types (numeric features, categorical label)
- Feature values fall within reasonable IRIS ranges
- Valid species labels
- Non-empty datasets

### Model Evaluation (`tests/test_model_evaluation.py`)
- Model loads and exposes `.predict()`
- Prediction count matches evaluation set size
- Accuracy, precision, recall, and F1 meet minimum thresholds
- Generates `metrics_report.md` consumed by the CML step

## Running Tests Locally

```bash
pip install -r requirements.txt
dvc pull
pytest tests/ -v
```

## Merging to Main

1. Push the `week_4` branch — CI runs automatically.
2. Open a pull request from `week_4` into `main`.
3. Confirm the CI check runs and the CML bot posts the metrics report as a PR comment.
4. Review results and merge.

## Notes

- Metric thresholds in `test_model_evaluation.py` (`MIN_ACCURACY`, `MIN_PRECISION`, `MIN_RECALL`, `MIN_F1`) should be tuned to match your actual model's expected performance.
- Adjust `data/`, `models/` paths in the test files if your repo layout differs.
