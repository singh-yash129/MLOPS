# Week 11 - Governing the Fine-Tuned LLM Guardrails on the IRIS Pipeline (Branch: `week_11`)

This repository contains the implementation of LLM-specific governance controls—specifically **Input and Output Guardrails**—designed to protect a fine-tuned Gemini model against prompt injection and context leakage attacks. 

It extends the traditional ML pipeline built in Week 10 by introducing MLSecOps practices tailored to the unique threat surface of Large Language Models.

## 🏗️ Architecture & Pipeline Flow

The pipeline intercepts and scans all data before it reaches the model and before it is returned to the user:

1. **User Input** ➔ 
2. **Input Guardrail** (Intercepts prompt injection & schema violations) ➔ 
3. **Fine-Tuned Gemini Model** (Vertex AI Endpoint) ➔ 
4. **Output Guardrail** (Filters context leakage & format violations) ➔ 
5. **Safe Response**

## 📂 Repository Structure

| File | Purpose |
| :--- | :--- |
| `guardrails.py` | Contains the `InputGuardrail`, `OutputGuardrail`, and `GuardedPipeline` classes. Implements regex pattern matching, structural schema validation, and exponential backoff for Vertex AI rate limits. |
| `red_team_suite.py` | Defines the adversarial datasets, containing 5 distinct prompt injection attacks and 5 prompt leakage probes used to red-team the pipeline. |
| `evaluate_guarded_pipeline.py` | The main evaluation script. It runs both adversarial and legitimate test sets through the guarded pipeline and computes block rates, false positive rates, and accuracy deltas. |
| `.github/workflows/ci-llm-governance.yml` | CI/CD automation that runs the evaluation script on every push. It enforces strict security thresholds (e.g., >90% block rate) and fails the build if the model regresses. |
| `data/` | Directory containing the legitimate test datasets from Week 10 (`iris_v1_raw_test.jsonl`, etc.) and the generated `governance_results.json` metrics. |

## 🛡️ Key Features

*   **Red-Team Evaluation:** Tests the pipeline against deliberate jailbreaks, instruction overrides, and context window extraction attempts.
*   **Input Sanitization:** Validates that incoming requests conform to the expected IRIS feature schema (numerical keys for v1, natural language for v2) and blocks malicious keywords.
*   **Output Filtering:** Scans model responses for sensitive fragments of the system prompt and ensures the final output strictly adheres to the required classification format.
*   **Rate-Limit Handling:** Integrates a robust exponential backoff and proactive 6-15 second pacing strategy to gracefully handle Google Cloud Vertex AI `429 Resource Exhausted` quota limits without crashing.

## 🚀 Setup and Execution

### Prerequisites
*   Google Cloud Platform (GCP) project with Vertex AI enabled.
*   Fine-tuned Gemini endpoints from Week 10.
*   Python 3.10+

### Local Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Authenticate with Google Cloud:

```bash
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
```


3. Run the evaluation script:
   ```bash
   python evaluate_guarded_pipeline.py \
       --v1-endpoint projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<V1_ENDPOINT_ID> \
       --v2-endpoint projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<V2_ENDPOINT_ID>

```

## 🔄 CI/CD Automation
This repository uses GitHub Actions for automated regression testing. The workflow authenticates to GCP using Workload Identity Federation (WIF).

To run successfully, the following GitHub Secrets must be configured:

WIF_PROVIDER

WIF_SERVICE_ACCOUNT

GCP_PROJECT_ID

V1_ENDPOINT_NAME

V2_ENDPOINT_NAME

  ## 📊 Evaluation Metrics
  The pipeline automatically calculates and saves the following metrics to data/governance_results.json:

  Injection Block Rate: The percentage of prompt injections successfully intercepted. (Target: ≥ 90%)

  Leakage Block Rate: The percentage of leakage attempts successfully intercepted. (Target: ≥ 90%)

  False Positive Rate: The percentage of legitimate, benign inputs incorrectly flagged by the guardrails. (Target: ≤ 5%)

  Accuracy Delta: The difference in classification accuracy between the unguarded and guarded pipeline.