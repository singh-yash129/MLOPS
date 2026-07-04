"""
Task 4: Fetch historical features from Feast's OFFLINE store and train the
IRIS classification model. Note that this script never reads the raw CSV
for features -- it pulls everything through Feast.
"""

import pandas as pd
from feast import FeatureStore
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "iris_features:sepal_length",
    "iris_features:sepal_width",
    "iris_features:petal_length",
    "iris_features:petal_width",
]


def main():
    store = FeatureStore(repo_path=".")

    # entity_df only needs the entity key + event_timestamp (+ label) --
    # NOT the feature columns themselves. Feast fetches those from the
    # offline store via point-in-time join.
    raw = pd.read_csv("data/iris_data_adapted_for_feast.csv")
    entity_df = raw[["iris_id", "event_timestamp", "species"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    print("Training dataframe pulled from Feast offline store:")
    print(training_df.head())
    print(f"Shape: {training_df.shape}")

    X = training_df[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y = training_df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    dump(model, "iris_model.joblib")
    print("Saved trained model to iris_model.joblib")


if __name__ == "__main__":
    main()