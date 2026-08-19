"""
task2_fairness_audit.py

Uses Fairlearn's MetricFrame to compute accuracy, precision, and recall
disaggregated by the `location` sensitive attribute, to check whether
the model performs equitably across the two groups.

Usage:
    python task2_fairness_audit.py
"""

import joblib
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def main():
    model = joblib.load("models/iris_model.joblib")
    df = pd.read_csv("data/iris_test_with_location.csv")

    X_test = df[FEATURE_COLS]
    y_test = df["species"]
    sensitive_features = df["location"]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    print("Overall metrics:")
    print(mf.overall)
    print("\nMetrics disaggregated by location group:")
    print(mf.by_group)

    print("\nMax difference between groups (per metric):")
    print(mf.difference(method="between_groups"))

    print("\nMax ratio between groups (per metric, closer to 1.0 = more equitable):")
    print(mf.ratio(method="between_groups"))

    mf.by_group.to_csv("data/fairness_metrics_by_group.csv")
    print("\nSaved -> data/fairness_metrics_by_group.csv")


if __name__ == "__main__":
    main()