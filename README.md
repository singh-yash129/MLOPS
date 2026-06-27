# IRIS ML Pipeline on Vertex AI with DVC

## Overview

This project implements an end-to-end Machine Learning pipeline for the IRIS dataset using Google Cloud Platform (GCP). It covers two weeks of MLOps implementation:

- **Week 1** — Training and inference pipeline on Vertex AI with GCS artifact storage
- **Week 2** — Data and model versioning using DVC backed by Google Cloud Storage

---

## Project Structure

```text
.
├── .dvc/
│   └── config                  # DVC remote configuration (GCS)
├── .dvcignore
│
├── src/                        # Week 1 — training and inference scripts
│   ├── train.py
│   └── inference.py
│
├── scripts/                    # Week 2 — DVC integrated training script
│   └── train.py
│
├── data/                       # Week 1 — versioned IRIS datasets
│   ├── raw/
│   │   └── iris.csv
│   ├── v1/
│   │   └── data.csv
│   └── v2/
│       └── data.csv
│
├── dvc_data/                   # Week 2 — DVC tracked datasets (pointers)
│   ├── iris_iter_1.csv.dvc
│   ├── iris_iter_2.csv.dvc
│   └── iris_iter_3.csv.dvc
│
├── dvc_models/                 # Week 2 — DVC tracked models (pointers)
│   ├── model_iter_1.pkl.dvc
│   ├── model_iter_2.pkl.dvc
│   ├── model_iter_3.pkl.dvc
│   ├── metrics_iter_1.json.dvc
│   ├── metrics_iter_2.json.dvc
│   └── metrics_iter_3.json.dvc
│
├── artifacts/                  # Week 1 — timestamped training artifacts
│   ├── v1/
│   │   ├── 20260614T051801/
│   │   └── 20260614T051908/
│   └── v2/
│       └── 20260614T052124/
│
├── requirements.txt
└── README.md
```

---

## Week 1 — Vertex AI Training & Inference Pipeline

### How It Works

- Downloads IRIS dataset from GCS
- Trains a Random Forest Classifier
- Saves model artifacts to timestamped GCS folders
- Runs inference and uploads evaluation results back to GCS

### GCS Bucket Structure (Week 1)

```text
gs://23f2004644-mlops/
├── data/
│   ├── v1/data.csv
│   └── v2/data.csv
└── artifacts/
    ├── v1/
    │   ├── 20260614T051801/
    │   │   ├── model.joblib
    │   │   ├── run_log.txt
    │   │   ├── eval_set.csv
    │   │   └── evaluation_results.txt
    │   └── 20260614T051908/
    │       ├── model.joblib
    │       ├── run_log.txt
    │       ├── eval_set.csv
    │       └── evaluation_results.txt
    └── v2/
        └── 20260614T052124/
            ├── model.joblib
            ├── run_log.txt
            ├── eval_set.csv
            └── evaluation_results.txt
```

### Training Execution

```bash
python src/train.py --bucket_name 23f2004644-mlops --data_version v1
```

### Inference Execution

```bash
python src/inference.py --bucket_name 23f2004644-mlops --artifact_prefix artifacts/v1/20260614T051801/
```

### Results

| Dataset Version | Accuracy |
|-----------------|----------|
| v1              | 0.9048   |
| v2              | 1.0000   |

---

## Week 2 — DVC Integration

### Why DVC?

Week 1 organized artifacts by timestamp but could not answer:
> *"Which exact version of data produced which model, and how do I go back to it?"*

DVC solves this by storing lightweight pointer files in Git while pushing actual data and models to GCS. This gives full version history and reproducibility.

### How DVC Works Here

```
Git commit  →  stores .dvc pointer files  (tiny ~200 bytes)
                        │
                        │ md5 hash reference
                        ▼
GCS Remote  →  stores actual .csv / .pkl files
```

### DVC Remote (GCS)

```
gs://23f2004644-mlops/dvc-store
```

### Dataset Iterations

| Iteration | Rows | Description                     |
|-----------|------|---------------------------------|
| 1         | 150  | Base IRIS dataset               |
| 2         | 200  | Base + 50 augmented rows        |
| 3         | 250  | Base + 100 augmented rows       |

### Running Week 2 Pipeline

```bash
# Install dependencies
pip install dvc "dvc[gs]" scikit-learn pandas numpy

# Pull all data and models from GCS
dvc pull

# Run a specific iteration
python scripts/train.py --iteration 1
python scripts/train.py --iteration 2
python scripts/train.py --iteration 3
```

### Switching Between Versions

```bash
# See all commits
git log --oneline

# Switch to iteration 1
git checkout <iter1-hash> -- dvc_data/iris_iter_1.csv.dvc dvc_models/model_iter_1.pkl.dvc
dvc checkout dvc_data/iris_iter_1.csv.dvc dvc_models/model_iter_1.pkl.dvc

# Return to latest
git checkout HEAD -- .
dvc checkout
```

---

## Requirements

```
pandas
numpy
scikit-learn
joblib
google-cloud-storage
dvc
dvc[gs]
```

---

## Learnings

**Week 1**
- Google Cloud Storage for dataset and artifact management
- Vertex AI Workbench setup and usage
- Training and inference pipeline separation
- Artifact versioning using timestamps

**Week 2**
- DVC setup and configuration with GCS remote
- Data and model versioning across multiple iterations
- Switching between data versions using Git + DVC checkout
- Reproducible ML pipelines without bloating Git history