"""
Task 6 (optional): Fetch the latest/best model from the MLflow Model Registry
and save it as a plain pickle file, so it can be bundled into the Docker
image without the running container needing MLflow at runtime.
"""

import argparse
import pickle

import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alias", default=None,
        help="Optional alias (e.g. 'champion') to fetch instead of latest version",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    if args.alias:
        model_uri = f"models:/{args.model_name}@{args.alias}"
    else:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(args.model_name)
        if not versions:
            raise RuntimeError(f"No registered versions found for '{args.model_name}'")
        latest_version = max(int(v.version) for v in versions)
        model_uri = f"models:/{args.model_name}/{latest_version}"

    print(f"Fetching model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    with open(args.output, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()