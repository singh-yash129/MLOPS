"""
prepare_v1_raw.py

Task 1 — Converts the IRIS dataset into v1 (raw feature) JSONL format
for Vertex AI supervised fine-tuning: each line is
{"input_text": "...", "output_text": "..."}
where input_text is the four raw feature values serialized as text and
output_text is just the species class name.

Produces both a training file and a held-out test file (test set is
reused by Task 4 evaluation and by the v2 script, via the same fixed
row-index split, so v1 and v2 are evaluated on the same underlying
samples for a fair comparison).

Usage:
    python prepare_v1_raw.py
"""

import json
import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "data"

SPECIES_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_iris_df() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species_idx"})
    df["species"] = df["species_idx"].map(SPECIES_NAMES)
    return df


def get_fixed_split(df: pd.DataFrame):
    """
    One fixed train/test split, stratified by species, reused by both
    v1 and v2 preparation scripts so the same physical rows end up in
    each version's test set -- this is what makes the Task 4 v1-vs-v2
    comparison a fair one.
    """
    idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["species"]
    )
    return train_idx, test_idx


def make_v1_record(row) -> dict:
    input_text = (
        f"sepal_length: {row['sepal length (cm)']}, "
        f"sepal_width: {row['sepal width (cm)']}, "
        f"petal_length: {row['petal length (cm)']}, "
        f"petal_width: {row['petal width (cm)']}"
    )
    return {"input_text": input_text, "output_text": row["species"]}


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_iris_df()
    train_idx, test_idx = get_fixed_split(df)

    train_records = [make_v1_record(df.loc[i]) for i in train_idx]
    test_records = [make_v1_record(df.loc[i]) for i in test_idx]

    train_path = f"{OUTPUT_DIR}/iris_v1_raw_train.jsonl"
    test_path = f"{OUTPUT_DIR}/iris_v1_raw_test.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    # Persist the split indices so v2 preparation reuses the exact same
    # rows for train/test.
    pd.Series(train_idx).to_csv(f"{OUTPUT_DIR}/split_train_idx.csv", index=False)
    pd.Series(test_idx).to_csv(f"{OUTPUT_DIR}/split_test_idx.csv", index=False)

    print(f"Saved {len(train_records)} training records -> {train_path}")
    print(f"Saved {len(test_records)} test records -> {test_path}")
    print("\nSample v1 training record:")
    print(json.dumps(train_records[0]))


if __name__ == "__main__":
    main()