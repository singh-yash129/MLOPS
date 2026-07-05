"""
Task 6 (step 1): Load iris_data_adapted_for_feast.csv into BigQuery.

Run this once, from a GCP environment authenticated with a project that
has BigQuery enabled (e.g. a GCP notebook instance, Colab with
`google.colab.auth.authenticate_user()`, or a local machine with
`gcloud auth application-default login` already run).

Usage:
    python load_to_bigquery.py <YOUR_GCP_PROJECT_ID>
"""

import sys

import pandas as pd
from google.cloud import bigquery


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_to_bigquery.py <YOUR_GCP_PROJECT_ID>")
        sys.exit(1)

    project_id = sys.argv[1]
    dataset_id = "iris_feast_dataset"
    table_id = "iris_features"

    client = bigquery.Client(project=project_id)

    # Create the dataset if it doesn't already exist
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)
    print(f"Dataset {project_id}.{dataset_id} ready")

    # Load and type the CSV the same way we did for the parquet file
    df = pd.read_csv("iris_data_adapted_for_feast.csv")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])
    df["iris_id"] = df["iris_id"].astype("int64")

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("iris_id", "INTEGER"),
            bigquery.SchemaField("sepal_length", "FLOAT"),
            bigquery.SchemaField("sepal_width", "FLOAT"),
            bigquery.SchemaField("petal_length", "FLOAT"),
            bigquery.SchemaField("petal_width", "FLOAT"),
            bigquery.SchemaField("species", "STRING"),
            bigquery.SchemaField("created_timestamp", "TIMESTAMP"),
        ],
    )

    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()  # wait for the load job to finish

    table = client.get_table(full_table_id)
    print(f"Loaded {table.num_rows} rows into {full_table_id}")


if __name__ == "__main__":
    main()