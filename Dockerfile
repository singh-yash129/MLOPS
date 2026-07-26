# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

WORKDIR /app

# System deps needed for scikit-learn / mlflow client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

COPY app/main.py .

# ---- Task 6 (optional): fetch the best model from the MLflow Model Registry
# at build time and bundle it into the image, so the running container never
# needs runtime access to the MLflow tracking server.
#
# Build with:
#   docker build \
#     --build-arg FETCH_MODEL=true \
#     --build-arg MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
#     --build-arg MODEL_NAME=iris_random_forest \
#     -t iris-api .
#
# If FETCH_MODEL is not set, the image is built without a model — mount one
# at runtime instead via the MODEL_PATH environment variable / a volume.
ARG FETCH_MODEL=false
ARG MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ARG MODEL_NAME=iris_random_forest

COPY mlflow.db* ./
COPY mlruns ./mlruns
COPY scripts/fetch_model_for_container.py ./scripts/fetch_model_for_container.py

RUN mkdir -p /app/model && \
    if [ "$FETCH_MODEL" = "true" ]; then \
        python scripts/fetch_model_for_container.py \
            --tracking-uri "$MLFLOW_TRACKING_URI" \
            --model-name "$MODEL_NAME" \
            --output /app/model/model.pkl ; \
    else \
        echo "FETCH_MODEL not set — building without a bundled model." ; \
    fi

ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]