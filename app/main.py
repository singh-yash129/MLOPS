"""
IRIS Inference API.

Serves predictions from a trained RandomForest model. By default it loads
the model bundled into the Docker image at build time (Task 6: fetched from
the MLflow Model Registry during the build). If no bundled model is present,
it falls back to loading from a local path for local development.
"""

import os
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

app = FastAPI(title="IRIS Inference API", version="1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Make sure the model was bundled during the Docker build."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., alias="sepal length (cm)")
    sepal_width: float = Field(..., alias="sepal width (cm)")
    petal_length: float = Field(..., alias="petal length (cm)")
    petal_width: float = Field(..., alias="petal width (cm)")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction: int
    species: str


SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.get("/")
def root():
    return {"status": "ok", "service": "iris-inference-api"}


@app.get("/health")
def health():
    """Readiness/liveness probe endpoint for Kubernetes."""
    try:
        get_model()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    model = get_model()
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    pred = int(model.predict(X)[0])
    return PredictionResponse(prediction=pred, species=SPECIES_MAP.get(pred, "unknown"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)