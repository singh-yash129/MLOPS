"""
task3_shap_analysis.py

Generates SHAP summary plots for all three IRIS classes (setosa,
versicolor, virginica) using a full-dataset TreeExplainer, and prints a
plain-language breakdown of the virginica plot specifically.

Usage:
    python task3_shap_analysis.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def main():
    import os
    os.makedirs("plots", exist_ok=True)

    model = joblib.load("models/iris_model.joblib")
    df = pd.read_csv("data/iris_with_location.csv")
    X = df[FEATURE_COLS]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values shape: (n_samples, n_features, n_classes) in recent
    # shap versions for multi-class sklearn models -- handle both that
    # and the older list-of-arrays format for compatibility.
    if isinstance(shap_values, list):
        per_class_values = shap_values
    else:
        per_class_values = [shap_values[:, :, i] for i in range(len(CLASS_NAMES))]

    for i, class_name in enumerate(CLASS_NAMES):
        plt.figure()
        shap.summary_plot(
            per_class_values[i], X, feature_names=FEATURE_COLS, show=False
        )
        plt.title(f"SHAP Summary — {class_name}")
        out_path = f"plots/shap_summary_{class_name}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved -> {out_path}")

    # Plain-language breakdown for virginica specifically (Task 3 focus)
    virginica_idx = CLASS_NAMES.index("virginica")
    virginica_shap = per_class_values[virginica_idx]
    mean_abs_shap = np.abs(virginica_shap).mean(axis=0)

    # Correlation between a feature's raw value and its own SHAP value
    # is what the plot's color-vs-position pattern actually encodes:
    # positive correlation = high (red) values sit on the right (push
    # toward virginica); negative correlation = high values sit on the
    # left (push away). Averaging signed SHAP across the whole dataset
    # would be misleading here since most rows are NOT virginica.
    correlations = [
        np.corrcoef(X[col].values, virginica_shap[:, j])[0, 1]
        for j, col in enumerate(FEATURE_COLS)
    ]

    print("\n--- Virginica SHAP breakdown (plain-language) ---")
    ranking = sorted(
        zip(FEATURE_COLS, mean_abs_shap, correlations),
        key=lambda t: -t[1],
    )
    for feature, mean_abs, corr in ranking:
        if corr > 0.3:
            reading = "HIGH values of this feature push TOWARD virginica (red dots on the right)"
        elif corr < -0.3:
            reading = "HIGH values of this feature push AWAY FROM virginica (red dots on the left)"
        else:
            reading = "no strong consistent direction (mixed effect)"
        print(f"{feature:22s} | importance (avg |SHAP|)={mean_abs:.4f} | {reading}")


if __name__ == "__main__":
    main()