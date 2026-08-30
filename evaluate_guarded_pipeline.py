"""
evaluate_guarded_pipeline.py - Task 5 Metrics & Effectiveness Evaluator
"""

import argparse
import json
import os
import time
import pandas as pd
from guardrails import GuardedPipeline, InputGuardrail, OutputGuardrail
from red_team_suite import ADVERSARIAL_PROMPT_INJECTION_SUITE, ADVERSARIAL_PROMPT_LEAKAGE_SUITE

def run_red_team_eval(pipeline: GuardedPipeline, attack_suite: list, category: str):
    blocked_count = 0
    results = []
    for item in attack_suite:
        inp = item["input"]
        pattern = item.get("attack_pattern") or item.get("leakage_pattern")
        res = pipeline.predict(inp)
        
        # PACING: Sleep to prevent Vertex AI 429 Resource Exhausted errors
        if not pipeline.mock:
            time.sleep(6)

        # A successful block occurs if input or output guardrails flag the attack
        is_blocked = res["status"] in ["BLOCKED_BY_INPUT_GUARDRAIL", "FILTERED_BY_OUTPUT_GUARDRAIL"]
        if is_blocked:
            blocked_count += 1
            
        results.append({
            "attack_pattern": pattern,
            "input": inp,
            "model_version": pipeline.version,
            "status": res["status"],
            "raw_response": res.get("raw_response"),
            "blocked": is_blocked,
        })
    block_rate = blocked_count / len(attack_suite)
    return block_rate, results

def evaluate_legitimate_test_set(pipeline: GuardedPipeline, test_file: str):
    records = [json.loads(line) for line in open(test_file)]
    total = len(records)
    false_positives = 0
    correct_predictions = 0

    for r in records:
        input_text = r["contents"][0]["parts"][0]["text"]
        expected = r["contents"][1]["parts"][0]["text"].strip().lower()
        if pipeline.version == "v2":
            expected = expected.replace("this is iris ", "").replace(".", "").strip()

        res = pipeline.predict(input_text)
        
        # PACING: Sleep to prevent Vertex AI 429 Resource Exhausted errors
        if not pipeline.mock:
            time.sleep(6)

        if res["status"] in ["BLOCKED_BY_INPUT_GUARDRAIL", "FILTERED_BY_OUTPUT_GUARDRAIL"]:
            false_positives += 1
        elif res.get("parsed_species") == expected:
            correct_predictions += 1

    fp_rate = false_positives / total
    guarded_accuracy = correct_predictions / total
    return fp_rate, guarded_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-endpoint", default="mock-v1")
    parser.add_argument("--v2-endpoint", default="mock-v2")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without GCP")
    args = parser.parse_args()

    v1_pipeline = GuardedPipeline(args.v1_endpoint, "v1", mock=args.mock)

    print("Running Injection Evaluation...")
    # 1. Run Injection Red-Teaming
    inj_block_rate, inj_details = run_red_team_eval(v1_pipeline, ADVERSARIAL_PROMPT_INJECTION_SUITE, "Injection")
    
    print("Running Leakage Evaluation...")
    # 2. Run Leakage Red-Teaming
    leak_block_rate, leak_details = run_red_team_eval(v1_pipeline, ADVERSARIAL_PROMPT_LEAKAGE_SUITE, "Leakage")

    print("Running False Positive Evaluation on Valid Test Set...")
    # 3. Measure False Positives & Accuracy Delta on Legitimate Data
    fp_rate, guarded_acc = evaluate_legitimate_test_set(v1_pipeline, "data/iris_v1_raw_test.jsonl")

    # Assuming baseline accuracy from Week 10 evaluation JSON if present
    baseline_acc = 0.0
    if os.path.exists("data/evaluation_comparison.json"):
        eval_data = json.load(open("data/evaluation_comparison.json"))
        # Fetch v1 baseline specifically
        for model_data in eval_data:
            if model_data["version"] == "v1":
                baseline_acc = model_data.get("accuracy", 0.0)

    accuracy_delta = baseline_acc - guarded_acc

    # Summary Metrics Table
    summary = {
        "Metric": ["Injection Block Rate", "Leakage Block Rate", "False Positive Rate", "Accuracy Delta"],
        "Ungoverned Pipeline": ["0.0%", "0.0%", "0.0%", "N/A"],
        "Guarded Pipeline": [f"{inj_block_rate*100:.1f}%", f"{leak_block_rate*100:.1f}%", f"{fp_rate*100:.1f}%", f"{accuracy_delta*100:.1f}%"],
    }

    print("\n" + "=" * 50)
    print("      GOVERNANCE EFFECTIVENESS REPORT")
    print("=" * 50)
    print(pd.DataFrame(summary).to_string(index=False))

    # Output to JSON for the GitHub Actions YAML to read
    os.makedirs("data", exist_ok=True)
    results_out = {
        "injection_block_rate": float(inj_block_rate),
        "leakage_block_rate": float(leak_block_rate),
        "false_positive_rate": float(fp_rate),
        "accuracy_delta": float(accuracy_delta)
    }
    
    with open("data/governance_results.json", "w") as f:
        json.dump(results_out, f, indent=4)
        
    print("\nMetrics successfully saved to data/governance_results.json for CI validation.")

if __name__ == "__main__":
    main()