# IRIS ML Pipeline on Vertex AI

## Overview

This project implements an end-to-end Machine Learning pipeline for the IRIS dataset using Google Cloud Platform (GCP). The pipeline uses Google Cloud Storage (GCS) for dataset and artifact storage and Vertex AI Workbench for training and inference execution.

## Project Structure

```text
.
├── data
│   ├── v1
│   └── v2
│
├── src
│   ├── train.py
│   └── inference.py
│
├── artifacts
│   ├── v1
│   └── v2
│
├── requirements.txt
└── README.md
```

## File Descriptions

### data/v1/data.csv

Version 1 of the IRIS dataset used for model training and evaluation.

### data/v2/data.csv

Version 2 of the IRIS dataset used to demonstrate data versioning and performance comparison.

### src/train.py

Training pipeline script that:

* Downloads dataset from Google Cloud Storage.
* Splits data into training and evaluation sets.
* Trains a Random Forest Classifier.
* Generates model artifacts.
* Uploads artifacts to timestamped folders in GCS.

Generated artifacts:

* model.joblib
* run_log.txt
* eval_set.csv

### src/inference.py

Inference pipeline script that:

* Downloads trained model and evaluation dataset from GCS.
* Performs predictions on evaluation data.
* Calculates accuracy and classification metrics.
* Uploads evaluation results back to GCS.


### artifacts/

This folder contains all output artifacts generated during training and inference. Artifacts are organized by dataset version and execution timestamp to ensure reproducibility and traceability.

Example structure:

```text
artifacts/
├── v1/
│   ├── 20260614T051801/
│   │   ├── model.joblib
│   │   ├── run_log.txt
│   │   ├── eval_set.csv
│   │   └── evaluation_results.txt
│   │
│   └── 20260614T051908/
│       ├── model.joblib
│       ├── run_log.txt
│       ├── eval_set.csv
│       └── evaluation_results.txt
│
└── v2/
    └── 20260614T052124/
        ├── model.joblib
        ├── run_log.txt
        ├── eval_set.csv
        └── evaluation_results.txt
```

#### Artifact Description

* **model.joblib**: Serialized trained Random Forest model.
* **run_log.txt**: Training run metadata including timestamp, dataset version, and training accuracy.
* **eval_set.csv**: Evaluation dataset used for inference.
* **evaluation_results.txt**: Accuracy score and classification report generated during inference.

The timestamp-based folder structure allows multiple training and inference executions to be tracked independently.


### requirements.txt

Contains all Python package dependencies required for running the project.

Examples:

* pandas
* scikit-learn
* joblib
* google-cloud-storage

### README.md

Project documentation containing setup instructions, file descriptions, execution steps, and results.

## GCS Bucket Structure

```text
gs://23f2004644-mlops/

data/
├── v1/data.csv
└── v2/data.csv

artifacts/
├── v1/
│   ├── 20260614T051801/
│   └── 20260614T051908/
└── v2/
    └── 20260614T052124/
```

## Training Execution

```bash
python src/train.py --bucket_name 23f2004644-mlops --data_version v1
```

## Inference Execution

```bash
python src/inference.py --bucket_name 23f2004644-mlops --artifact_prefix artifacts/v1/20260614T051801/
```

## Results

| Dataset Version | Accuracy |
| --------------- | -------- |
| v1              | 0.9048   |
| v2              | 1.0000   |

## Learnings

* Google Cloud Storage for dataset and artifact management.
* Vertex AI Workbench setup and usage.
* Training and inference pipeline separation.
* Artifact versioning using timestamps.
* Data versioning and model performance comparison.
* Basic MLOps workflow implementation on GCP.
