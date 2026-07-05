"""
Task 5: Simulate real-time inference. Given iris_id(s), fetch features
from Feast's ONLINE store (not the CSV) and pass them to the trained model.
Then compare against predictions made directly from the raw CSV to prove
there is no training/serving skew.
"""

import pandas as pd
from feast import FeatureStore
from joblib import load

FEATURES = [
    "iris_features:sepal_length",
    "iris_features:sepal_width",
    "iris_features:petal_length",
    "iris_features:petal_width",
]

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def predict_from_online_store(store, model, iris_ids):
    online = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"iris_id": i} for i in iris_ids],
    ).to_dict()

    df = pd.DataFrame(online)
    X = df[FEATURE_COLS]
    preds = model.predict(X)
    return df.assign(prediction=preds)


def predict_from_raw_csv(model, iris_ids, csv_path="data/iris_data_adapted_for_feast.csv"):
    raw = pd.read_csv(csv_path)
    # take the most recent row per iris_id, mirroring what materialize()
    # pushed into the online store
    latest = (
        raw.sort_values("event_timestamp")
        .groupby("iris_id")
        .tail(1)
        .set_index("iris_id")
        .loc[iris_ids]
        .reset_index()
    )
    X = latest[FEATURE_COLS]
    preds = model.predict(X)
    return latest.assign(prediction=preds)


def main():
    store = FeatureStore(repo_path=".")
    model = load("iris_model.joblib")

    iris_ids = [1001, 1002, 1003]

    online_result = predict_from_online_store(store, model, iris_ids)
    print("Predictions using ONLINE store (simulated real-time inference):")
    print(online_result[["iris_id"] + FEATURE_COLS + ["prediction"]])

    raw_result = predict_from_raw_csv(model, iris_ids)
    print("\nPredictions using raw CSV directly (for comparison):")
    print(raw_result[["iris_id"] + FEATURE_COLS + ["prediction"]])

    match = list(online_result.sort_values("iris_id")["prediction"]) == list(
        raw_result.sort_values("iris_id")["prediction"]
    )
    print(f"\nPredictions match (no training/serving skew): {match}")


if __name__ == "__main__":
    main()