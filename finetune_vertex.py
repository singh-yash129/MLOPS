"""
finetune_vertex.py

Task 3 — Submits TWO Vertex AI supervised fine-tuning jobs for a
cost-efficient Gemini model: one on the v1 (raw feature) dataset, one
on the v2 (natural language description) dataset. Both jobs use
identical hyperparameters so the only variable between them is the
data representation.

PREREQUISITES (run once, outside this script):
    gcloud auth application-default login
    gcloud config set project <YOUR_PROJECT_ID>
    pip install google-cloud-aiplatform

    # Upload the JSONL files produced by prepare_v1_raw.py and
    # prepare_v2_description.py to GCS first:
    gsutil cp data/iris_v1_raw_train.jsonl gs://<YOUR_BUCKET>/llmops/iris_v1_raw_train.jsonl
    gsutil cp data/iris_v2_description_train.jsonl gs://<YOUR_BUCKET>/llmops/iris_v2_description_train.jsonl

NOTE: This script actually submits billable Vertex AI tuning jobs. It
cannot be executed in a sandboxed/offline environment -- it requires a
real GCP project with Vertex AI API enabled and billing configured.
Run it from your own authenticated environment (Cloud Shell, local
gcloud, or your VM).

Usage:
    python finetune_vertex.py
"""

import time

from vertexai.preview.tuning import sft
import vertexai

import os

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
BASE_MODEL = "gemini-2.5-flash-lite-001"

BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME")
if not PROJECT_ID or not BUCKET_NAME:
    raise EnvironmentError(
        "Set GCP_PROJECT_ID and GCP_BUCKET_NAME environment variables "
        "before running this script (see README for instructions)."
    )
BUCKET = f"gs://{BUCKET_NAME}/llmops"
V1_TRAIN_URI = f"{BUCKET}/iris_v1_raw_train.jsonl"
V2_TRAIN_URI = f"{BUCKET}/iris_v2_description_train.jsonl"

# Same hyperparameters for both jobs -- data representation is the
# only variable being tested.
TUNING_HYPERPARAMS = dict(
    epochs=3,
    learning_rate_multiplier=1.0,
)


def submit_tuning_job(display_name: str, train_dataset_uri: str):
    print(f"Submitting tuning job: {display_name}")
    print(f"  base model:   {BASE_MODEL}")
    print(f"  training set: {train_dataset_uri}")
    print(f"  hyperparams:  {TUNING_HYPERPARAMS}")

    sft_tuning_job = sft.train(
        source_model=BASE_MODEL,
        train_dataset=train_dataset_uri,
        tuned_model_display_name=display_name,
        epochs=TUNING_HYPERPARAMS["epochs"],
        learning_rate_multiplier=TUNING_HYPERPARAMS["learning_rate_multiplier"],
    )
    return sft_tuning_job


def wait_for_completion(job, label: str, poll_seconds: int = 60):
    print(f"\nWaiting for {label} tuning job to complete "
          f"(this can take significant time)...")
    while not job.has_ended:
        time.sleep(poll_seconds)
        job.refresh()
        print(f"  [{label}] state: {job.state}")
    print(f"[{label}] finished. state={job.state}")
    return job


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    v1_job = submit_tuning_job("iris-v1-raw-tuned", V1_TRAIN_URI)
    v2_job = submit_tuning_job("iris-v2-description-tuned", V2_TRAIN_URI)

    print(f"\nv1 job resource name: {v1_job.resource_name}")
    print(f"v2 job resource name: {v2_job.resource_name}")
    print("\nBoth jobs submitted. Track progress in the Vertex AI console:")
    print(f"https://console.cloud.google.com/vertex-ai/generative/language/tuning?project={PROJECT_ID}")

    # Optional: block and poll until both finish, then print the tuned
    # endpoint resource names needed for evaluate_and_compare.py
    v1_job = wait_for_completion(v1_job, "v1-raw")
    v2_job = wait_for_completion(v2_job, "v2-description")

    print("\n--- Tuned model endpoints (save these for evaluation) ---")
    print(f"v1 tuned endpoint: {v1_job.tuned_model_endpoint_name}")
    print(f"v2 tuned endpoint: {v2_job.tuned_model_endpoint_name}")


if __name__ == "__main__":
    main()
