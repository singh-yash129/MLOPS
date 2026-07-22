"""
Week 5: Model Evaluation Tests (MLflow version)
Loads the latest registered model from MLflow Model Registry,
runs inference, and asserts key metrics meet minimum thresholds.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "iris_random_forest"

MIN_ACCURACY = 0.90
MIN_PRECISION = 0.85
MIN_RECALL = 0.85
MIN_F1 = 0.85


@pytest.fixture(scope="module")
def model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(REGISTERED_MODEL_NAME)
    assert latest, f"No registered versions found for '{REGISTERED_MODEL_NAME}'"
    latest_version = max(int(v.version) for v in latest)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version}"
    return mlflow.sklearn.load_model(model_uri)


@pytest.fixture(scope="module")
def eval_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y


@pytest.fixture(scope="module")
def predictions(model, eval_data):
    X, y_true = eval_data
    y_pred = model.predict(X)
    return y_true, y_pred


def test_model_loads_from_registry(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_model_accuracy(predictions):
    y_true, y_pred = predictions
    acc = accuracy_score(y_true, y_pred)
    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.4f} below threshold {MIN_ACCURACY}"


def test_model_precision(predictions):
    y_true, y_pred = predictions
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    assert precision >= MIN_PRECISION, f"Precision {precision:.4f} below threshold {MIN_PRECISION}"


def test_model_recall(predictions):
    y_true, y_pred = predictions
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    assert recall >= MIN_RECALL, f"Recall {recall:.4f} below threshold {MIN_RECALL}"


def test_model_f1(predictions):
    y_true, y_pred = predictions
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert f1 >= MIN_F1, f"F1 {f1:.4f} below threshold {MIN_F1}"