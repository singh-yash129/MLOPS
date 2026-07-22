"""
Week 5 - Task 5: Fetch model from MLflow Model Registry for evaluation.

Loads the latest (or a specific alias/stage) version of the registered
model directly from MLflow, instead of reading a DVC-tracked pickle file.
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

REGISTERED_MODEL_NAME = "iris_random_forest"


def load_model(tracking_uri: str, version: str = None, alias: str = None):
    """
    Load a model from the MLflow Model Registry.

    - If `version` is given, loads that exact registered version (e.g. "3").
    - If `alias` is given, loads the version currently tagged with that
      alias (e.g. "champion").
    - If neither is given, loads the latest registered version.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if alias:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    elif version:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
    else:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(REGISTERED_MODEL_NAME)
        if not latest:
            raise RuntimeError(
                f"No registered versions found for model '{REGISTERED_MODEL_NAME}'. "
                "Run training first."
            )
        latest_version = max(int(v.version) for v in latest)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version}"

    print(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri", type=str, default="sqlite:///mlflow.db",
        help="MLflow tracking URI",
    )
    parser.add_argument("--version", type=str, default=None, help="Specific model version")
    parser.add_argument("--alias", type=str, default=None, help="Model alias, e.g. 'champion'")
    args = parser.parse_args()

    model = load_model(args.tracking_uri, version=args.version, alias=args.alias)

    # Evaluate against the base IRIS dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro", zero_division=0)
    recall = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    main()