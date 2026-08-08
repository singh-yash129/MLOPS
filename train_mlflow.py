"""
train_mlflow.py

Trains an IRIS classifier on the clean dataset and each poisoned variant
(5%, 10%, 50%), logging every run to MLflow with the poisoning level as
a parameter and accuracy/precision/recall/F1 as metrics.

Usage:
    mlflow ui   # in a separate terminal, to view results at http://127.0.0.1:5000
    python train_mlflow.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
EXPERIMENT_NAME = "iris-data-poisoning"
RANDOM_SEED = 42

DATASETS = {
    0: f"{DATA_DIR}/iris_clean.csv",
    5: f"{DATA_DIR}/iris_poisoned_5pct.csv",
    10: f"{DATA_DIR}/iris_poisoned_10pct.csv",
    50: f"{DATA_DIR}/iris_poisoned_50pct.csv",
}

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def get_fixed_split_indices():
    """
    Compute ONE train/test row-index split, stratified on the ORIGINAL
    clean labels, and reuse it across every dataset variant. This is
    critical: all CSVs share the same row order, so applying the same
    index split to each means every run trains/tests on the same
    physical rows -- the only thing that differs between runs is
    whether those training rows were poisoned, not which rows were
    picked. Without this, sklearn's stratify=y (where y itself has
    been corrupted) can silently select different train/test rows per
    dataset, making cross-run comparisons meaningless.
    """
    clean_df = pd.read_csv(DATASETS[0])
    idx = clean_df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED,
        stratify=clean_df["species"],
    )
    return train_idx, test_idx


def load_and_split(path: str, train_idx):
    df = pd.read_csv(path)
    X_train = df.loc[train_idx, FEATURE_COLS]
    y_train = df.loc[train_idx, "species"]
    return X_train, y_train


def load_clean_test_set(test_idx):
    """Always evaluate against the same clean, held-out test rows."""
    df = pd.read_csv(DATASETS[0])
    X_test = df.loc[test_idx, FEATURE_COLS]
    y_test = df.loc[test_idx, "species"]
    return X_test, y_test


def train_and_log(poison_pct: int, data_path: str, train_idx, X_test, y_test):
    X_train, y_train = load_and_split(data_path, train_idx)

    with mlflow.start_run(run_name=f"poison_{poison_pct}pct"):
        mlflow.log_param("poisoning_level_pct", poison_pct)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("model_type", "RandomForestClassifier")

        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{poison_pct}% poisoned] acc={acc:.3f} prec={prec:.3f} "
              f"rec={rec:.3f} f1={f1:.3f}")


def main():
    # Newer MLflow versions deprecate the plain-folder file store in
    # favor of a database backend -- sqlite is the simplest local
    # option and is what `mlflow ui --backend-store-uri sqlite:///mlflow.db`
    # expects too.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_idx, test_idx = get_fixed_split_indices()
    X_test, y_test = load_clean_test_set(test_idx)

    for poison_pct, path in DATASETS.items():
        train_and_log(poison_pct, path, train_idx, X_test, y_test)

    print("\nAll runs logged. Launch `mlflow ui` and open the "
          f"'{EXPERIMENT_NAME}' experiment to compare runs.")


if __name__ == "__main__":
    main()