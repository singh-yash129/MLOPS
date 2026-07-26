"""
Task 6 (optional): Fetch the latest/best model from the MLflow Model Registry
and save it as a plain pickle file, so it can be bundled into the Docker
image without the running container needing MLflow at runtime.

Note: resolving models:/<name>/<version> URIs against a file-based
(non-server) MLflow backend can be unreliable -- the registry's internal
source bookkeeping doesn't always translate correctly outside a real
MLflow tracking server. To avoid this, we look up the model version's
underlying run_id and load the model via runs:/<run_id>/model instead,
which resolves directly against the run's own artifact location.
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
    client = mlflow.tracking.MlflowClient()

    if args.alias:
        version_info = client.get_model_version_by_alias(args.model_name, args.alias)
    else:
        versions = client.get_latest_versions(args.model_name)
        if not versions:
            raise RuntimeError(f"No registered versions found for '{args.model_name}'")
        version_info = max(versions, key=lambda v: int(v.version))

    run_id = version_info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Resolved model version {version_info.version} -> run_id={run_id}")
    print(f"Fetching model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    with open(args.output, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()