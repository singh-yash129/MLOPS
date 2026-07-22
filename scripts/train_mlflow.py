"""
Week 5 - Task 1 & 2: Hyperparameter tuning + MLflow experiment tracking.

Runs multiple training configurations (varying n_estimators and max_depth),
logs each run's parameters, metrics, and the trained model to MLflow,
and registers the resulting model in the MLflow Model Registry.

Data is still versioned through DVC (Task 4: only data, not models).
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

os.makedirs("dvc_data", exist_ok=True)

EXPERIMENT_NAME = "iris_classification"
REGISTERED_MODEL_NAME = "iris_random_forest"

# Realistic bounds for each IRIS feature (cm), used to clip augmentation noise
FEATURE_BOUNDS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.1, 3.5),
}


def clip_to_bounds(df, feature_names):
    for col in feature_names:
        low, high = FEATURE_BOUNDS[col]
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def load_data(iteration: int):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    if iteration >= 2:
        rng = np.random.RandomState(42)
        extra = df.sample(50, random_state=42).copy()
        extra[iris.feature_names] += rng.normal(0, 0.05, extra[iris.feature_names].shape)
        extra = clip_to_bounds(extra, iris.feature_names)
        df = pd.concat([df, extra], ignore_index=True)

    if iteration >= 3:
        rng = np.random.RandomState(99)
        extra = df.sample(100, random_state=99).copy()
        extra[iris.feature_names] += rng.normal(0, 0.05, extra[iris.feature_names].shape)
        extra = clip_to_bounds(extra, iris.feature_names)
        df = pd.concat([df, extra], ignore_index=True)

    data_path = f"dvc_data/iris_iter_{iteration}.csv"
    df.to_csv(data_path, index=False)
    return df, iris.feature_names, data_path


def run_experiment(df, feature_names, n_estimators, max_depth, iteration, data_path):
    X = df[feature_names].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}"):
        # Task 2: log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("rows", len(df))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Task 2: log evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Task 2: log the model as an MLflow artifact + register it
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"n_estimators={n_estimators}, max_depth={max_depth} -> "
            f"accuracy={acc:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}"
        )
        return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (local file store by default)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, feature_names, data_path = load_data(args.iteration)

    # Task 1: hyperparameter tuning - vary n_estimators and max_depth
    # across multiple training runs
    hyperparameter_grid = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": None},
    ]

    results = []
    for params in hyperparameter_grid:
        acc = run_experiment(
            df, feature_names,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            iteration=args.iteration,
            data_path=data_path,
        )
        results.append({**params, "accuracy": acc})

    best = max(results, key=lambda r: r["accuracy"])
    print("\nBest configuration:")
    print(best)


if __name__ == "__main__":
    main()