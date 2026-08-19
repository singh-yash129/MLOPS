"""
task4_drift_detection.py

Simulates a "production" IRIS dataset by shifting feature distributions
(a constant offset applied to petal length, plus a smaller shift to
petal width, leaving sepal features untouched), then statistically
compares training vs. simulated-production distributions per feature
using the Kolmogorov-Smirnov two-sample test.

Usage:
    python task4_drift_detection.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
RANDOM_SEED = 42
ALPHA = 0.05  # significance threshold for the KS test


def simulate_production_data(train_df: pd.DataFrame) -> pd.DataFrame:
    prod_df = train_df.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    # Simulate real-world drift: petal length shifts up by a constant
    # offset (e.g. measurement/environment change), petal width gets a
    # smaller shift plus noise, sepal features are left alone as a
    # built-in negative control.
    prod_df["petal length (cm)"] = prod_df["petal length (cm)"] + 1.5
    prod_df["petal width (cm)"] = prod_df["petal width (cm)"] + 0.4 + rng.normal(0, 0.1, len(prod_df))
    # sepal length (cm) and sepal width (cm) intentionally untouched
    return prod_df


def main():
    import os
    os.makedirs("plots", exist_ok=True)

    train_df = pd.read_csv("data/iris_with_location.csv")
    prod_df = simulate_production_data(train_df)
    prod_df.to_csv("data/iris_simulated_production.csv", index=False)
    print("Saved simulated production dataset -> data/iris_simulated_production.csv")

    print(f"\n{'Feature':22s} | {'Train mean':>10s} | {'Prod mean':>10s} | {'KS stat':>8s} | {'p-value':>10s} | Drift?")
    print("-" * 90)

    results = []
    for col in FEATURE_COLS:
        train_vals = train_df[col].values
        prod_vals = prod_df[col].values
        ks_stat, p_value = stats.ks_2samp(train_vals, prod_vals)
        drifted = p_value < ALPHA
        results.append((col, train_vals.mean(), prod_vals.mean(), ks_stat, p_value, drifted))
        flag = "YES - DRIFTED" if drifted else "no"
        print(f"{col:22s} | {train_vals.mean():10.3f} | {prod_vals.mean():10.3f} | "
              f"{ks_stat:8.3f} | {p_value:10.6f} | {flag}")

        # Save a distribution comparison plot per feature
        plt.figure(figsize=(6, 4))
        plt.hist(train_vals, bins=20, alpha=0.5, label="Training (original)", density=True)
        plt.hist(prod_vals, bins=20, alpha=0.5, label="Simulated production", density=True)
        plt.title(f"Distribution shift — {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        safe_name = col.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = f"plots/drift_{safe_name}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()

    results_df = pd.DataFrame(
        results, columns=["feature", "train_mean", "prod_mean", "ks_statistic", "p_value", "drifted"]
    )
    results_df.to_csv("data/drift_test_results.csv", index=False)
    print("\nSaved -> data/drift_test_results.csv")
    print("Saved per-feature distribution plots -> plots/drift_*.png")


if __name__ == "__main__":
    main()