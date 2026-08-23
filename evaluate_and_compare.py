"""
evaluate_and_compare.py

Task 4 — Runs both fine-tuned models (v1-raw, v2-description) against
their held-out test sets, computes accuracy, per-class precision/recall,
and format compliance rate, then prints a side-by-side comparison.

Format compliance: a response counts as compliant only if it maps
cleanly to exactly one of the three valid species names. Anything else
(JSON, extra commentary, hallucinated species, empty string) counts as
non-compliant -- this is an LLM-specific failure mode a traditional
classifier never has.

Usage:
    # Real evaluation against live Vertex AI endpoints:
    python evaluate_and_compare.py \\
        --v1-endpoint projects/.../locations/.../endpoints/... \\
        --v2-endpoint projects/.../locations/.../endpoints/...

    # Dry run with simulated predictions, to sanity-check the metric
    # computation logic without calling any live endpoint:
    python evaluate_and_compare.py --mock
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score

VALID_SPECIES = ["setosa", "versicolor", "virginica"]


def init_vertexai():
    """Required before any GenerativeModel/predict_vertex call."""
    import vertexai
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project_id:
        raise EnvironmentError(
            "Set GCP_PROJECT_ID before running evaluation against real "
            "endpoints, e.g.: export GCP_PROJECT_ID=$(gcloud config get-value project)"
        )
    vertexai.init(project=project_id, location=location)


# --------------------------------------------------------------------
# Real Vertex AI prediction calls (used when NOT running --mock)
# --------------------------------------------------------------------
def predict_vertex(endpoint_name: str, input_text: str, max_retries: int = 5) -> str:
    """Call a deployed Vertex AI tuned-model endpoint for one prediction.

    Retries on 429 RESOURCE_EXHAUSTED with exponential backoff -- tuned
    endpoints frequently have low per-minute quota (especially on trial
    projects), and a fresh deployment can also take a few minutes to
    fully warm up before it can serve steady traffic.
    """
    import time
    from google.api_core.exceptions import ResourceExhausted
    from vertexai.generative_models import GenerativeModel

    model = GenerativeModel(endpoint_name)

    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(input_text)
            return response.text.strip()
        except ResourceExhausted:
            if attempt == max_retries:
                raise
            print(f"  [429 quota hit] retrying in {delay}s "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(delay)
            delay *= 2  # exponential backoff: 5, 10, 20, 40, 80s



# --------------------------------------------------------------------
# Parsing model output into a predicted species label
# --------------------------------------------------------------------
def extract_species_v1(raw_output: str):
    """v1 expects the bare species name as output_text."""
    cleaned = raw_output.strip().strip('"').strip(".").lower()
    if cleaned in VALID_SPECIES:
        return cleaned
    return None  # non-compliant


def extract_species_v2(raw_output: str):
    """v2 expects a sentence like 'This is Iris virginica.'"""
    lowered = raw_output.lower()
    for species in VALID_SPECIES:
        if re.search(rf"\biris\s+{species}\b", lowered):
            return species
    for species in VALID_SPECIES:
        if re.search(rf"\b{species}\b", lowered):
            return species
    return None  # non-compliant


# --------------------------------------------------------------------
# Mock predictor -- lets us verify metric computation without a live
# endpoint. Simulates realistic behavior: mostly correct, some noise,
# and a few malformed non-compliant outputs.
# --------------------------------------------------------------------
def mock_predict(true_species: str, version: str, rng, error_rate=0.10, malformed_rate=0.05):
    roll = rng.random()
    if roll < malformed_rate:
        return "I believe this could possibly be a type of flower, perhaps setosa??"
    elif roll < malformed_rate + error_rate:
        wrong = rng.choice([s for s in VALID_SPECIES if s != true_species])
        chosen = wrong
    else:
        chosen = true_species

    if version == "v1":
        return chosen
    else:
        return f"This is Iris {chosen}."


# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------
def evaluate(test_path: str, version: str, endpoint_name: str, mock: bool, rng=None):
    records = [json.loads(line) for line in open(test_path)]

    true_labels = []
    pred_labels = []      # None where non-compliant

    extractor = extract_species_v1 if version == "v1" else extract_species_v2

    for r in records:
        # Test JSONL uses the same contents/role/parts schema as training
        # data -- extract the user prompt and the model's expected answer.
        input_text = r["contents"][0]["parts"][0]["text"]
        expected_output_text = r["contents"][1]["parts"][0]["text"]

        true_species = expected_output_text.strip().lower()
        if version == "v2":
            true_species = extract_species_v2(true_species)

        if mock:
            raw = mock_predict(true_species, version, rng)
        else:
            raw = predict_vertex(endpoint_name, input_text)
            time.sleep(1)  # small pacing delay between calls to avoid
                            # tripping per-minute quota limits

        pred = extractor(raw)

        true_labels.append(true_species)
        pred_labels.append(pred)

    n_total = len(true_labels)
    compliant_mask = [p is not None for p in pred_labels]
    n_compliant = sum(compliant_mask)
    format_compliance = n_compliant / n_total

    # Treat non-compliant predictions as simply incorrect (a distinct
    # predicted label "NONE") rather than dropping them -- dropping
    # would inflate accuracy by ignoring the LLM's failure to follow
    # format.
    pred_labels_filled = [p if p is not None else "NONE" for p in pred_labels]

    accuracy = sum(
        t == p for t, p in zip(true_labels, pred_labels_filled)
    ) / n_total

    labels_for_metrics = VALID_SPECIES
    precision = precision_score(
        true_labels, pred_labels_filled, labels=labels_for_metrics,
        average=None, zero_division=0
    )
    recall = recall_score(
        true_labels, pred_labels_filled, labels=labels_for_metrics,
        average=None, zero_division=0
    )

    return {
        "version": version,
        "n_total": n_total,
        "n_compliant": n_compliant,
        "format_compliance": format_compliance,
        "accuracy": accuracy,
        "precision_per_class": dict(zip(labels_for_metrics, precision)),
        "recall_per_class": dict(zip(labels_for_metrics, recall)),
    }


def print_report(result: dict):
    print(f"\n=== {result['version']} ===")
    print(f"Test samples:       {result['n_total']}")
    print(f"Format compliance:  {result['n_compliant']}/{result['n_total']} "
          f"({result['format_compliance']*100:.1f}%)")
    print(f"Accuracy:           {result['accuracy']*100:.1f}%")
    print("Per-class precision:")
    for cls, val in result["precision_per_class"].items():
        print(f"  {cls:12s}: {val:.3f}")
    print("Per-class recall:")
    for cls, val in result["recall_per_class"].items():
        print(f"  {cls:12s}: {val:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-endpoint", default=None,
                         help="Vertex AI tuned endpoint resource name for v1")
    parser.add_argument("--v2-endpoint", default=None,
                         help="Vertex AI tuned endpoint resource name for v2")
    parser.add_argument("--mock", action="store_true",
                         help="Use simulated predictions instead of calling live endpoints")
    parser.add_argument("--v1-test", default="data/iris_v1_raw_test.jsonl")
    parser.add_argument("--v2-test", default="data/iris_v2_description_test.jsonl")
    args = parser.parse_args()

    if not args.mock and (not args.v1_endpoint or not args.v2_endpoint):
        parser.error("--v1-endpoint and --v2-endpoint are required unless --mock is set")

    if not args.mock:
        init_vertexai()

    import numpy as np
    rng = np.random.default_rng(42)

    v1_result = evaluate(args.v1_test, "v1", args.v1_endpoint, args.mock, rng)
    v2_result = evaluate(args.v2_test, "v2", args.v2_endpoint, args.mock, rng)

    print_report(v1_result)
    print_report(v2_result)

    print("\n=== v1 vs v2 comparison ===")
    print(f"{'Metric':22s} | {'v1 (raw)':>12s} | {'v2 (description)':>18s}")
    print("-" * 58)
    print(f"{'Accuracy':22s} | {v1_result['accuracy']*100:11.1f}% | {v2_result['accuracy']*100:17.1f}%")
    print(f"{'Format compliance':22s} | {v1_result['format_compliance']*100:11.1f}% | {v2_result['format_compliance']*100:17.1f}%")

    winner = "v2 (description)" if v2_result["accuracy"] > v1_result["accuracy"] else \
             "v1 (raw)" if v1_result["accuracy"] > v2_result["accuracy"] else "tie"
    print(f"\nHigher accuracy: {winner}")

    pd.DataFrame([v1_result, v2_result]).to_json(
        "data/evaluation_comparison.json", orient="records", indent=2
    )
    print("\nSaved -> data/evaluation_comparison.json")


if __name__ == "__main__":
    main()