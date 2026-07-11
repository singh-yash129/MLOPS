"""
Task 2: Model Evaluation Tests
Loads the trained model, runs inference on the evaluation set,
and asserts that key metrics meet minimum quality thresholds.
"""

import pickle

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "models/model.pkl"
EVAL_PATH = "data/eval.csv"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"

# Minimum acceptable thresholds — tune to your actual model's expected performance
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.85
MIN_RECALL = 0.85
MIN_F1 = 0.85


@pytest.fixture(scope="module")
def model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def eval_data():
    return pd.read_csv(EVAL_PATH)


@pytest.fixture(scope="module")
def predictions(model, eval_data):
    X_eval = eval_data[FEATURE_COLUMNS]
    y_true = eval_data[TARGET_COLUMN]
    y_pred = model.predict(X_eval)
    return y_true, y_pred


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict"), "Loaded model has no predict() method"


def test_model_predicts_expected_shape(eval_data, predictions):
    y_true, y_pred = predictions
    assert len(y_pred) == len(eval_data), "Prediction count does not match eval set size"


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
    assert f1 >= MIN_F1, f"F1 score {f1:.4f} below threshold {MIN_F1}"


def test_write_metrics_report(predictions):
    """
    Writes metrics to a file so CML can pick it up and post as a PR comment (Task 5).
    """
    y_true, y_pred = predictions
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    with open("metrics_report.md", "w") as f:
        f.write("## Model Evaluation Report\n\n")
        f.write("| Metric | Value | Threshold | Status |\n")
        f.write("|--------|-------|-----------|--------|\n")
        f.write(f"| Accuracy | {acc:.4f} | {MIN_ACCURACY} | {'✅' if acc >= MIN_ACCURACY else '❌'} |\n")
        f.write(f"| Precision | {precision:.4f} | {MIN_PRECISION} | {'✅' if precision >= MIN_PRECISION else '❌'} |\n")
        f.write(f"| Recall | {recall:.4f} | {MIN_RECALL} | {'✅' if recall >= MIN_RECALL else '❌'} |\n")
        f.write(f"| F1 Score | {f1:.4f} | {MIN_F1} | {'✅' if f1 >= MIN_F1 else '❌'} |\n")

    assert True  # this test just generates the report artifact