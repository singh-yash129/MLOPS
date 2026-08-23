"""
debug_raw_outputs.py

Quick standalone diagnostic -- calls each tuned endpoint on just 3
test samples and prints the RAW, unprocessed text response, so we can
see exactly what the model is actually returning before deciding how
to fix the parsing/extraction logic.

Usage:
    export GCP_PROJECT_ID=$(gcloud config get-value project)
    python3 debug_raw_outputs.py \
        --v1-endpoint projects/.../endpoints/... \
        --v2-endpoint projects/.../endpoints/...
"""

import argparse
import json
import os

import vertexai
from vertexai.generative_models import GenerativeModel

N_SAMPLES = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-endpoint", required=True)
    parser.add_argument("--v2-endpoint", required=True)
    args = parser.parse_args()

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise EnvironmentError("Set GCP_PROJECT_ID first: export GCP_PROJECT_ID=$(gcloud config get-value project)")
    vertexai.init(project=project_id, location=os.environ.get("GCP_LOCATION", "us-central1"))

    for version, endpoint, test_file in [
        ("v1", args.v1_endpoint, "data/iris_v1_raw_test.jsonl"),
        ("v2", args.v2_endpoint, "data/iris_v2_description_test.jsonl"),
    ]:
        print(f"\n{'='*20} {version} ({endpoint}) {'='*20}")
        model = GenerativeModel(endpoint)
        records = [json.loads(line) for line in open(test_file)][:N_SAMPLES]

        for i, r in enumerate(records):
            input_text = r["contents"][0]["parts"][0]["text"]
            expected = r["contents"][1]["parts"][0]["text"]

            print(f"\n--- Sample {i+1} ---")
            print(f"INPUT:    {input_text}")
            print(f"EXPECTED: {expected!r}")

            response = model.generate_content(input_text)
            print(f"RAW RESPONSE (repr): {response.text!r}")

            # Also print finish_reason / safety info in case the
            # response was truncated or blocked rather than just chatty
            try:
                candidate = response.candidates[0]
                print(f"finish_reason: {candidate.finish_reason}")
            except Exception as e:
                print(f"(could not read finish_reason: {e})")


if __name__ == "__main__":
    main()
