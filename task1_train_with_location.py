"""
task1_train_with_location.py

Adds a randomly-assigned `location` sensitive attribute (0 or 1) to the
IRIS dataset. The classifier is trained ONLY on the original four
numeric features -- location is never used as a training feature, only
as a group label for later fairness auditing (Task 2).

Usage:
    python task1_train_with_location.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def build_dataset_with_location() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})

    rng = np.random.default_rng(RANDOM_SEED)
    # location is a random group label, NOT correlated with species or
    # features -- it exists purely to demonstrate the fairness-audit
    # workflow on a known-fair split.
    df["location"] = rng.integers(0, 2, size=len(df))  # 0 or 1
    return df


def main():
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = build_dataset_with_location()
    df.to_csv("data/iris_with_location.csv", index=False)
    print(f"Saved dataset with location attribute -> data/iris_with_location.csv")
    print(df["location"].value_counts().rename("row_count"))

    X = df[FEATURE_COLS]           # location excluded from training features
    y = df["species"]
    loc = df["location"]

    X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
        X, y, loc, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\nOverall test accuracy: {acc:.3f}")

    joblib.dump(model, "models/iris_model.joblib")
    X_test.assign(species=y_test, location=loc_test).to_csv(
        "data/iris_test_with_location.csv", index=False
    )
    print("Saved trained model -> models/iris_model.joblib")
    print("Saved test set (with location) -> data/iris_test_with_location.csv")


if __name__ == "__main__":
    main()