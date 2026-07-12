import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs("dvc_data", exist_ok=True)
os.makedirs("dvc_models", exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    # Load base IRIS dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Augment data based on iteration
    if args.iteration == 2:
        rng = np.random.RandomState(42)
        extra = df.sample(50, random_state=42).copy()
        extra[iris.feature_names] += rng.normal(0, 0.05, extra[iris.feature_names].shape)
        df = pd.concat([df, extra], ignore_index=True)
        print(f"Iteration 2: Added 50 rows → total {len(df)} rows")

    elif args.iteration == 3:
        rng = np.random.RandomState(99)
        extra = df.sample(100, random_state=99).copy()
        extra[iris.feature_names] += rng.normal(0, 0.05, extra[iris.feature_names].shape)
        df = pd.concat([df, extra], ignore_index=True)
        print(f"Iteration 3: Added 100 rows → total {len(df)} rows")

    else:
        print(f"Iteration 1: Base dataset → {len(df)} rows")

    # Save dataset
    data_path = f"dvc_data/iris_iter_{args.iteration}.csv"
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path}")

    # Train model
    X = df[iris.feature_names].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # Save model
    model_path = f"dvc_models/model_iter_{args.iteration}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save metrics
    metrics = {"iteration": args.iteration, "rows": len(df), "accuracy": round(acc, 4)}
    metrics_path = f"dvc_models/metrics_iter_{args.iteration}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()