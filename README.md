# IRIS Inference API — Continuous Deployment (Week 6)

Containerizes the IRIS inference API with Docker and deploys it to Google Kubernetes Engine (GKE), with the entire build → push → deploy cycle automated through GitHub Actions.

This branch is self-contained — it does not depend on any files from earlier weekly branches. Training, containerization, and deployment are all defined here.

## Pipeline Overview

```
Code Push
    │
    ▼
CI (tests pass, from Week 4)
    │
    ▼
CD Pipeline
    │
    ├── Train + register model (MLflow)
    ├── Docker Build        (package API + model image)
    ├── Artifact Registry   (push container image)
    └── GKE                 (deploy to Kubernetes)
    │
    ▼
Live API
```

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── cd.yml                       # Train, build, push, deploy — end to end
├── app/
│   ├── main.py                          # FastAPI inference service
│   └── requirements.txt                 # API runtime dependencies
├── scripts/
│   ├── train_for_deployment.py          # Standalone training + MLflow registration
│   └── fetch_model_for_container.py     # Pulls the registered model at build time
├── k8s/
│   ├── deployment.yaml                  # Kubernetes Deployment (2 replicas, health probes)
│   └── service.yaml                     # LoadBalancer Service exposing port 80 → 8080
├── Dockerfile                           # Builds the API image, optionally bundles the model
├── requirements.txt                     # Training/build-time dependencies
└── README.md
```

## Pod vs Container (Task 1)

A **Docker container** is a single running process with its own isolated filesystem — the packaging and runtime unit for one application.

A **Kubernetes Pod** wraps one or more containers that are always scheduled together on the same node, sharing the same network namespace (same IP, reachable via `localhost` between containers) and storage volumes.

Kubernetes deploys Pods rather than raw containers because it needs an atomic unit of scheduling — a group of tightly-coupled containers (e.g., a main app plus a logging sidecar) that must be placed, scaled, and restarted together. The Pod is that atomic unit; the container is what actually runs inside it.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Readiness/liveness probe — confirms the model is loaded |
| `POST` | `/predict` | Runs inference; accepts the four IRIS feature measurements |

Example request:
```bash
curl -X POST http://<EXTERNAL_IP>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }'
```
Response:
```json
{"prediction": 0, "species": "setosa"}
```

## GCP Setup (Task 3)

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud services enable artifactregistry.googleapis.com container.googleapis.com --project="$PROJECT_ID"

gcloud artifacts repositories create iris-repo \
  --repository-format=docker --location=us-central1 --project="$PROJECT_ID"

gcloud iam service-accounts create github-cd-deploy \
  --project="$PROJECT_ID" --display-name="GitHub Actions CD"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:github-cd-deploy@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:github-cd-deploy@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"

gcloud container clusters create-auto iris-cluster \
  --region=us-central1 --project="$PROJECT_ID"
```

Authentication uses **Workload Identity Federation** (no downloaded service account key, per org policy) — reusing the same WIF pool/provider pattern established in Week 4, bound to this service account instead.

Required GitHub repository secrets:

| Secret | Value |
|---|---|
| `WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `WIF_SERVICE_ACCOUNT_CD` | `github-cd-deploy@<PROJECT_ID>.iam.gserviceaccount.com` |

## Workflow (`.github/workflows/cd.yml`)

Triggers on every push, plus manual `workflow_dispatch`. Steps:

1. Checkout, authenticate to GCP via WIF
2. **Task 6** — train a fresh model and register it in MLflow (`scripts/train_for_deployment.py`)
3. **Task 2 & 4** — build the Docker image, fetching the registered model at build time (`scripts/fetch_model_for_container.py`) and bundling it in
4. Push the image to Artifact Registry, tagged with both the commit SHA and `latest`
5. **Task 5** — fetch GKE credentials, apply the Deployment and Service manifests, wait for rollout
6. Verify the LoadBalancer IP is assigned and call `/health` to confirm the API is live

## Running Locally

```bash
pip install -r requirements.txt
python3 scripts/train_for_deployment.py

docker build --build-arg FETCH_MODEL=true -t iris-api .
docker run -p 8080:8080 iris-api

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}'
```

## Notes

- `FETCH_MODEL=false` (the Dockerfile's default) builds the API image without bundling a model — useful for testing the container structure without needing MLflow at build time.
- The MLflow tracking backend used here is SQLite (`sqlite:///mlflow.db`), consistent with Week 5 — the plain filesystem store is in maintenance mode in the MLflow version used.
- This branch was deliberately kept independent of `week_2`–`week_5` (no DVC, no Feast, no CI test files) so the CD pipeline can be evaluated and demonstrated on its own.