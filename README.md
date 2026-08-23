# Week 10 — From MLOps to LLMOps: Fine-Tuning Gemini on the IRIS Pipeline

Branch: `week_10`

## Setup (run in GCP Cloud Shell, after cloning this repo)

```bash
git checkout week_10
pip install -r requirements.txt

gcloud config set project <PROJECT_ID>
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

## Run all tasks in order

```bash
# Task 1 — v1 raw feature JSONL
python3 prepare_v1_raw.py

# Task 2 — v2 natural language description JSONL (reuses v1's train/test split)
python3 prepare_v2_description.py

# Upload both datasets to GCS
gsutil mb -l us-central1 gs://<BUCKET_NAME>
gsutil cp data/iris_v1_raw_train.jsonl gs://<BUCKET_NAME>/llmops/
gsutil cp data/iris_v2_description_train.jsonl gs://<BUCKET_NAME>/llmops/

# Task 3 — submit both fine-tuning jobs (edit PROJECT_ID/BUCKET at top of the file first)
python3 finetune_vertex.py

# Task 4 — evaluate both tuned models once jobs complete
python3 evaluate_and_compare.py \
  --v1-endpoint <V1_ENDPOINT_NAME> \
  --v2-endpoint <V2_ENDPOINT_NAME>

# Dry-run the evaluation logic without a live endpoint (sanity check only):
python3 evaluate_and_compare.py --mock
```

Task 5 (optional) — `ci-llm-eval.yml` extends GitHub Actions to run evaluation
automatically on push and fail the build if either model's accuracy drops
below the threshold set in the workflow.

## Files

| File | Purpose |
|---|---|
| `prepare_v1_raw.py` | Task 1 — raw feature JSONL, fixed train/test split |
| `prepare_v2_description.py` | Task 2 — natural language JSONL, same split as v1 |
| `finetune_vertex.py` | Task 3 — submits both Vertex AI fine-tuning jobs |
| `evaluate_and_compare.py` | Task 4 — accuracy, per-class precision/recall, format compliance |
| `ci-llm-eval.yml` | Task 5 (optional) — CI regression guard on push |

## Generated output

```
data/
  iris_v1_raw_train.jsonl
  iris_v1_raw_test.jsonl
  iris_v2_description_train.jsonl
  iris_v2_description_test.jsonl
  split_train_idx.csv
  split_test_idx.csv
  evaluation_comparison.json
```

## Notes on verification

`prepare_v1_raw.py` and `prepare_v2_description.py` were run and verified locally —
120 train / 30 test records each, confirmed both versions map to identical
underlying rows. `evaluate_and_compare.py`'s metric logic was verified using
`--mock` mode. `finetune_vertex.py` requires a real GCP project with billing
and Vertex AI quota — it cannot be run outside GCP.

## Submission

```
<IITM_BS_ID>Assignment10<TERM><YEAR>_MLOps.<File_Type>
```