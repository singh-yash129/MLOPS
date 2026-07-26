"""
Week 6 - standalone model training for the CD pipeline.

Trains a simple RandomForest on the base IRIS dataset and registers it in
MLflow, so Task 6 (bundling the model into the Docker image) has something
to fetch. This script has no dependency on any previous week's branch,
DVC, or Feast — it's self-contained for week_6.
"""

import argparse

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "iris_classification"
REGISTERED_MODEL_NAME = "iris_random_forest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri", type=str, default="sqlite:///mlflow.db",
        help="MLflow tracking URI (SQLite backend by default)",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"cd_rf_n{args.n_estimators}_d{args.max_depth}"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        print(f"Registered model '{REGISTERED_MODEL_NAME}' — ready for container bundling.")


if __name__ == "__main__":
    main()