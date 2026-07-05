# Week 3 — Integrating Feast Feature Store into the IRIS Pipeline

This repo implements a Feast feature store layer on top of the IRIS
classification pipeline, so that training and inference read features
from a single consistent source instead of each re-reading/re-deriving
them from the raw CSV.

Dataset: `iris_data_adapted_for_feast.csv` — a time-aware version of the
IRIS dataset with 3 tracked plants (`iris_id` 1001, 1002, 1003), each with
15 days of measurements, real `event_timestamp` and `created_timestamp`
columns. Source: `IITMBSMLOps/ga_resources` repo, branch `week_3`.

---

## Repository Structure

```
23F2004644_MLOPS_WEEKLY_ASSIGNMENT/
├── Feast_IRIS_Assignment_Week3.ipynb
├── Modified_Driver_Ranking_Tutorial.ipynb
├── LICENSE
├── README.md
├── requirements.txt
├── iris_data_adapted_for_feast.csv
├── load_to_bigquery.py
├── iris_feature_repo/
│   ├── feature_store.yaml
│   ├── iris_repo.py
│   ├── parquet_converter.py
│   ├── train.py
│   ├── inference.py
│   ├── iris_model.joblib
│   └── data/
└── iris_feature_repo_bq/
    ├── feature_store.yaml
    ├── iris_repo_bq.py
    └── data/
```

The `data/` folders are populated automatically by Feast itself when you
run `feast apply` (creates the registry) and `feast materialize`
(creates/updates the online store) — their exact contents depend on
your Feast version and aren't hand-authored files, so they aren't
enumerated file-by-file here.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Task 1 — Initialize the Feast Feature Repository

The Feast repository lives in `iris_feature_repo/` — a `feature_store.yaml`
plus a `data/` directory, following Feast's conventions directly rather
than using `feast init` (which scaffolds a nested subfolder along with
unrelated example data not relevant to this dataset).

`feature_store.yaml` uses a local provider with a SQLite online store —
no cloud dependency for Tasks 1–5.

## Task 2 — Entities, Data Sources & Feature Views

Defined in `iris_feature_repo/iris_repo.py`:
- **Entity**: `iris_id` — uniquely identifies each iris plant.
- **Data source**: `FileSource` pointing at the Parquet-converted
  dataset, with `timestamp_field="event_timestamp"` set explicitly
  (Feast can't auto-infer it here since both `event_timestamp` and
  `created_timestamp` look like timestamp columns).
- **Feature view**: `iris_features`, mapping `sepal_length`,
  `sepal_width`, `petal_length`, `petal_width`, `species` to the entity
  and source.

The CSV is converted to Parquet once using `parquet_converter.py`
(`FileSource` is most reliable against Parquet):

```bash
cd iris_feature_repo
python3 parquet_converter.py --input data/iris_data_adapted_for_feast.csv --output data/iris_data_adapted_for_feast.parquet
```

## Task 3 — Apply & Materialize

```bash
cd iris_feature_repo
feast apply
feast materialize 2025-09-01T00:00:00 2025-10-05T00:00:00
```

`feast apply` registers the entity/feature view in the registry.
`feast materialize` populates the SQLite online store for the given date
range — confirmed by "Materializing 1 feature views ... iris_features"
in the console output.

## Task 4 — Offline Retrieval for Training

```bash
cd iris_feature_repo
python3 train.py
```

Pulls historical features via `store.get_historical_features()` — the
entity dataframe only supplies `iris_id`, `event_timestamp`, and the
label (`species`); Feast performs the point-in-time join to attach
feature values. Trains a `RandomForestClassifier` and saves it to
`iris_model.joblib`.

## Task 5 — Online Retrieval for Inference

```bash
cd iris_feature_repo
python3 inference.py
```

Fetches features for given `iris_id`s from the **online** store via
`store.get_online_features()`, predicts with the trained model, and
compares those predictions against predictions made directly from the
raw CSV — demonstrating no training/serving skew.

## Task 6 — BigQuery Backend

```bash
python3 load_to_bigquery.py <YOUR_GCP_PROJECT_ID>
```
Loads the dataset into `iris_feast_dataset.iris_features` in BigQuery.

Then, inside `iris_feature_repo_bq/feature_store.yaml` and
`iris_repo_bq.py`, replace `<YOUR_GCP_PROJECT_ID>` with your actual
project ID, and run:

```bash
cd iris_feature_repo_bq
feast apply
feast materialize 2025-09-01T00:00:00 2025-10-05T00:00:00
```

This swaps only the **offline store** to BigQuery (the online store
stays SQLite, since BigQuery isn't built for millisecond-level serving
reads). Retrieval code is unchanged — Feast abstracts the backend
behind `get_historical_features()`.

**Trade-offs vs local SQLite/file backend:**
- **Latency**: BigQuery queries pay network + query-planning overhead
  (hundreds of ms to seconds) vs near-instant local Parquet reads.
- **Cost**: billed per byte scanned; negligible for this 45-row dataset,
  but a real consideration at production scale.
- **Scalability**: BigQuery scales to far larger datasets and concurrent
  training jobs than a local file/SQLite setup can.
- **Setup overhead**: requires GCP project, IAM roles
  (`BigQuery Data Editor`, `BigQuery Job User`), and authenticated
  network access — more moving parts than a local file.

---

## Requirements

- Run in a GCP environment (per assignment rules).
- GCP service account / user needs BigQuery Data Editor + BigQuery Job
  User roles for Task 6.
