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

## Contents

| File | Purpose |
|---|---|
| `Feast_IRIS_Assignment_Week3.ipynb` | Full walkthrough notebook — Tasks 1–6, runs end to end |
| `feature_store.yaml` | Local/SQLite Feast config (Tasks 1–5) |
| `iris_repo.py` | Entity, data source, feature view definitions (Tasks 1–2), local `FileSource` |
| `train.py` | Task 4 — trains the classifier using Feast's **offline** store |
| `inference.py` | Task 5 — simulates real-time inference using Feast's **online** store, compares against raw-CSV predictions |
| `feature_store_bigquery.yaml` | Task 6 — Feast config with BigQuery as the offline store |
| `iris_repo_bq.py` | Task 6 — same entity/feature view, `BigQuerySource` instead of `FileSource` |
| `load_to_bigquery.py` | Task 6 — one-off script to load the CSV into a BigQuery table |

---

## Setup

```bash
pip install feast scikit-learn pandas google-cloud-bigquery
```

Place `iris_data_adapted_for_feast.csv` in the project root (same folder
as these files).

---

## Task 1 — Initialize the Feast Feature Repository

```bash
mkdir -p iris_feature_repo/data
cp feature_store.yaml iris_feature_repo/feature_store.yaml
cp iris_repo.py iris_feature_repo/iris_repo.py
cp iris_data_adapted_for_feast.csv iris_feature_repo/data/
```

`feature_store.yaml` uses a local provider with a SQLite online store —
no cloud dependency for Tasks 1–5.

## Task 2 — Entities, Data Sources & Feature Views

Defined in `iris_repo.py`:
- **Entity**: `iris_id` — uniquely identifies each iris plant.
- **Data source**: `FileSource` pointing at the (Parquet-converted)
  dataset, with `timestamp_field="event_timestamp"` set explicitly
  (Feast can't auto-infer it here since both `event_timestamp` and
  `created_timestamp` look like timestamp columns).
- **Feature view**: `iris_features`, mapping `sepal_length`,
  `sepal_width`, `petal_length`, `petal_width`, `species` to the entity
  and source.

The CSV needs a one-time conversion to Parquet (`FileSource` is most
reliable with Parquet) — done in the notebook / see `train.py`'s data
prep step.

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
python train.py
```

Pulls historical features via `store.get_historical_features()` — the
entity dataframe only supplies `iris_id`, `event_timestamp`, and the
label (`species`); Feast performs the point-in-time join to attach
feature values. Trains a `RandomForestClassifier` and saves it to
`iris_model.joblib`.

## Task 5 — Online Retrieval for Inference

```bash
python inference.py
```

Fetches features for given `iris_id`s from the **online** store via
`store.get_online_features()`, predicts with the trained model, and
compares those predictions against predictions made directly from the
raw CSV — demonstrating no training/serving skew.

## Task 6 (Optional) — BigQuery Backend

```bash
python load_to_bigquery.py <YOUR_GCP_PROJECT_ID>
```
Loads the dataset into `iris_feast_dataset.iris_features` in BigQuery.

```bash
mkdir -p iris_feature_repo_bq/data
cp feature_store_bigquery.yaml iris_feature_repo_bq/feature_store.yaml
cp iris_repo_bq.py iris_feature_repo_bq/iris_repo_bq.py
# edit both files: replace <YOUR_GCP_PROJECT_ID> with your actual project ID

cd iris_feature_repo_bq
feast apply
feast materialize 2025-09-01T00:00:00 2025-10-05T00:00:00
```

This swaps only the **offline store** to BigQuery (the online store
stays SQLite, since BigQuery isn't built for millisecond-level serving
reads). `train.py`-equivalent retrieval code is unchanged — Feast
abstracts the backend behind `get_historical_features()`.

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

- Run in a GCP environment (per assignment rules) — required for Task 6,
  optional for Tasks 1–5.
- GCP service account / user needs BigQuery Data Editor + BigQuery Job
  User roles for Task 6.