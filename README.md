# IRIS Inference API - Stress Testing & Observability (Week 7)

This branch builds upon the Continuous Deployment pipeline by introducing **Stress Testing, Auto-scaling, and Observability**. It containerizes the IRIS inference API with Docker, deploys it to Google Kubernetes Engine (GKE) via GitHub Actions, and subjects the live endpoint to high-concurrency load testing using `wrk` while dynamically scaling via the Kubernetes Horizontal Pod Autoscaler (HPA).

## Pipeline Overview
`Code Push` ➔ `CI` ➔ `CD (Build -> Push -> Deploy)` ➔ **`Live API`** ➔ **`Stress Test & Monitor`**

* **GitHub Actions:** Builds the image, deploys to GKE, and automatically runs a `wrk` stress test against the newly provisioned External IP.
* **wrk:** Generates high-concurrency traffic (1,000+ connections) to find latency inflection points and system limits.
* **HPA (Horizontal Pod Autoscaler):** Automatically scales API pods up and down based on CPU utilization metrics.
* **GCP Observability:** Uses Cloud Monitoring and Cloud Logging to visualize pod-level resource consumption and trace API request bottlenecks.

## Repository Structure

```text
.github/
  workflows/
    cd.yml                       # Train, build, push, deploy, and STRESS TEST
app/
  main.py                        # FastAPI inference service
  requirements.txt               # API runtime dependencies
scripts/
  train_for_deployment.py        # Standalone training + MLflow registration
  fetch_model_for_container.py   # Pulls the registered model at build time
  post.lua                       # Lua script providing the JSON payload for wrk
k8s/
  deployment.yaml                # Kubernetes Deployment (with CPU/Memory requests & limits)
  service.yaml                   # LoadBalancer Service
  hpa.yaml                       # Horizontal Pod Autoscaler (min: 1, max: 3)
Dockerfile                       # Builds the API image, optionally bundles the model
requirements.txt                 # Training/build-time dependencies