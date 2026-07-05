"""
Convert iris_data_adapted_for_feast.csv into a Parquet file, with the
correct dtypes applied, so it can be used as a Feast FileSource.

Feast's FileSource works most reliably against Parquet rather than CSV
directly, and Parquet preserves timestamp/int/float types exactly
instead of relying on inference at read time.

Usage:
    python parquet_converter.py
    python parquet_converter.py --input iris_data_adapted_for_feast.csv --output data/iris_data_adapted_for_feast.parquet
"""

import argparse
import os

import pandas as pd


def convert_csv_to_parquet(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # Ensure timestamp columns are proper datetimes, not strings
    for col in ("event_timestamp", "created_timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Ensure entity key is a clean integer type
    if "iris_id" in df.columns:
        df["iris_id"] = df["iris_id"].astype("int64")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Converted {input_path} -> {output_path}")
    print(f"Rows: {len(df)}")
    print(df.dtypes)


def main():
    parser = argparse.ArgumentParser(description="Convert IRIS CSV to Parquet for Feast")
    parser.add_argument(
        "--input",
        default="iris_data_adapted_for_feast.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output",
        default="data/iris_data_adapted_for_feast.parquet",
        help="Path to write the output Parquet file",
    )
    args = parser.parse_args()

    convert_csv_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main()