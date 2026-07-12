"""
Task 1: Data Validation Tests
Validates schema, missing values, feature types, and value ranges
for the IRIS dataset (latest DVC-tracked iteration).
"""

import pandas as pd
import pytest

# Using the latest iteration as both the train and eval reference set.
# Update these paths if you want to validate a different iteration.
TRAIN_PATH = "dvc_data/iris_iter_3.csv"
EVAL_PATH = "dvc_data/iris_iter_3.csv"

EXPECTED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
]

NUMERIC_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

TARGET_COLUMN = "target"

# sklearn's load_iris() encodes species as integers: 0=setosa, 1=versicolor, 2=virginica
VALID_TARGET_VALUES = {0, 1, 2}

# Reasonable IRIS value ranges (cm), with a little slack
VALUE_RANGES = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.5),
}


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def eval_df():
    return pd.read_csv(EVAL_PATH)


# ---------- Schema ----------

def test_train_schema(train_df):
    missing_cols = set(EXPECTED_COLUMNS) - set(train_df.columns)
    assert not missing_cols, f"Missing columns in train data: {missing_cols}"


def test_eval_schema(eval_df):
    missing_cols = set(EXPECTED_COLUMNS) - set(eval_df.columns)
    assert not missing_cols, f"Missing columns in eval data: {missing_cols}"


# ---------- Missing values ----------

def test_train_no_missing_values(train_df):
    null_counts = train_df[EXPECTED_COLUMNS].isnull().sum()
    assert null_counts.sum() == 0, f"Missing values found:\n{null_counts[null_counts > 0]}"


def test_eval_no_missing_values(eval_df):
    null_counts = eval_df[EXPECTED_COLUMNS].isnull().sum()
    assert null_counts.sum() == 0, f"Missing values found:\n{null_counts[null_counts > 0]}"


# ---------- Feature types ----------

def test_train_feature_types(train_df):
    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(train_df[col]), (
            f"Column '{col}' should be numeric, got {train_df[col].dtype}"
        )
    assert pd.api.types.is_numeric_dtype(train_df[TARGET_COLUMN]), (
        "target column should be numeric (0, 1, or 2)"
    )


def test_eval_feature_types(eval_df):
    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(eval_df[col]), (
            f"Column '{col}' should be numeric, got {eval_df[col].dtype}"
        )


# ---------- Value ranges ----------

def test_train_value_ranges(train_df):
    for col, (low, high) in VALUE_RANGES.items():
        out_of_range = train_df[(train_df[col] < low) | (train_df[col] > high)]
        assert out_of_range.empty, (
            f"{len(out_of_range)} rows in train '{col}' fall outside [{low}, {high}]"
        )


def test_eval_value_ranges(eval_df):
    for col, (low, high) in VALUE_RANGES.items():
        out_of_range = eval_df[(eval_df[col] < low) | (eval_df[col] > high)]
        assert out_of_range.empty, (
            f"{len(out_of_range)} rows in eval '{col}' fall outside [{low}, {high}]"
        )


# ---------- Label validity ----------

def test_train_target_values_valid(train_df):
    unique_targets = set(train_df[TARGET_COLUMN].unique())
    assert unique_targets.issubset(VALID_TARGET_VALUES), (
        f"Unexpected target labels found: {unique_targets - VALID_TARGET_VALUES}"
    )


def test_eval_target_values_valid(eval_df):
    unique_targets = set(eval_df[TARGET_COLUMN].unique())
    assert unique_targets.issubset(VALID_TARGET_VALUES), (
        f"Unexpected target labels found: {unique_targets - VALID_TARGET_VALUES}"
    )


# ---------- Row count sanity ----------

def test_train_not_empty(train_df):
    assert len(train_df) > 0, "Training data is empty"


def test_eval_not_empty(eval_df):
    assert len(eval_df) > 0, "Evaluation data is empty"