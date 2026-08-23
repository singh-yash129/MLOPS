"""
prepare_v2_description.py

Task 2 — Converts the IRIS dataset into v2 (natural language
description) JSONL format for Vertex AI supervised fine-tuning: each
line is {"input_text": "...", "output_text": "..."} where input_text is
a full natural-language sentence describing the flower's measurements,
and output_text is a complete sentence naming the species.

Reuses the EXACT same train/test row split as prepare_v1_raw.py (via
the saved split index CSVs) so v1 and v2 are evaluated on the same
underlying samples in Task 4.

Usage:
    python prepare_v1_raw.py          # must run first, to generate the split
    python prepare_v2_description.py
"""

import json
import os

import pandas as pd
from sklearn.datasets import load_iris

OUTPUT_DIR = "data"
SPECIES_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}
SPECIES_SENTENCE_NAMES = {
    "setosa": "Iris setosa",
    "versicolor": "Iris versicolor",
    "virginica": "Iris virginica",
}


def load_iris_df() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species_idx"})
    df["species"] = df["species_idx"].map(SPECIES_NAMES)
    return df


def make_v2_record(row) -> dict:
    input_text = (
        f"A flower specimen has a sepal length of {row['sepal length (cm)']} cm, "
        f"sepal width of {row['sepal width (cm)']} cm, "
        f"petal length of {row['petal length (cm)']} cm, "
        f"and petal width of {row['petal width (cm)']} cm. "
        f"Identify the iris species."
    )
    output_text = f"This is {SPECIES_SENTENCE_NAMES[row['species']]}."
    return {"input_text": input_text, "output_text": output_text}


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    split_train_path = f"{OUTPUT_DIR}/split_train_idx.csv"
    split_test_path = f"{OUTPUT_DIR}/split_test_idx.csv"
    if not (os.path.exists(split_train_path) and os.path.exists(split_test_path)):
        raise FileNotFoundError(
            "Run prepare_v1_raw.py first -- it generates the shared "
            "train/test split that this script reuses."
        )

    train_idx = pd.read_csv(split_train_path)["0"].tolist()
    test_idx = pd.read_csv(split_test_path)["0"].tolist()

    df = load_iris_df()
    train_records = [make_v2_record(df.loc[i]) for i in train_idx]
    test_records = [make_v2_record(df.loc[i]) for i in test_idx]

    train_path = f"{OUTPUT_DIR}/iris_v2_description_train.jsonl"
    test_path = f"{OUTPUT_DIR}/iris_v2_description_test.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    print(f"Saved {len(train_records)} training records -> {train_path}")
    print(f"Saved {len(test_records)} test records -> {test_path}")
    print("\nSample v2 training record:")
    print(json.dumps(train_records[0]))


if __name__ == "__main__":
    main()