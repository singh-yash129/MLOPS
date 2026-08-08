"""
poison_data.py

Creates poisoned variants of the IRIS dataset for MLSecOps Week 8.

Poisoning method: for each sample selected for corruption, replace all
four feature values with random values (drawn from a plausible range)
and assign a random class label. This simulates an attacker injecting
noise into the training set.

Usage:
    python poison_data.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

RANDOM_SEED = 42
OUTPUT_DIR = "data"
POISON_LEVELS = [0.05, 0.10, 0.50]  # 5%, 10%, 50%

# Feature ranges based on the real IRIS dataset's observed min/max,
# widened slightly so poisoned values are plausible-looking but wrong.
FEATURE_RANGES = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.0, 3.5),
}


def load_clean_iris() -> pd.DataFrame:
    """Load the IRIS dataset as a clean, labeled DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns={"target": "species"})
    # species is stored as integer class 0/1/2 — keep it that way for
    # consistency with how we'll assign random poisoned labels below.
    return df


def poison_dataset(df: pd.DataFrame, poison_fraction: float, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Return a copy of df where `poison_fraction` of rows have been
    replaced with random feature values and a random label.
    """
    rng = np.random.default_rng(seed)
    poisoned_df = df.copy()

    n_samples = len(df)
    n_poison = int(round(n_samples * poison_fraction))
    poison_indices = rng.choice(n_samples, size=n_poison, replace=False)

    feature_cols = [c for c in df.columns if c != "species"]
    class_labels = sorted(df["species"].unique())

    for idx in poison_indices:
        for col in feature_cols:
            low, high = FEATURE_RANGES[col]
            poisoned_df.at[idx, col] = round(rng.uniform(low, high), 2)
        poisoned_df.at[idx, "species"] = rng.choice(class_labels)

    poisoned_df["is_poisoned"] = 0
    poisoned_df.loc[poison_indices, "is_poisoned"] = 1

    return poisoned_df


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_df = load_clean_iris()
    clean_df["is_poisoned"] = 0
    clean_path = f"{OUTPUT_DIR}/iris_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved clean baseline dataset -> {clean_path} ({len(clean_df)} rows)")

    for level in POISON_LEVELS:
        poisoned_df = poison_dataset(clean_df.drop(columns=["is_poisoned"]), level)
        pct = int(level * 100)
        out_path = f"{OUTPUT_DIR}/iris_poisoned_{pct}pct.csv"
        poisoned_df.to_csv(out_path, index=False)
        n_poisoned = poisoned_df["is_poisoned"].sum()
        print(f"Saved {pct}% poisoned dataset -> {out_path} "
              f"({n_poisoned}/{len(poisoned_df)} rows corrupted)")


if __name__ == "__main__":
    main()