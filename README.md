# MLOps Graded Assignment - Week 3 Resources 

# Time-Aware Iris Dataset for Feast Tutorial

## Overview

This directory contains a modified, time-series version of the classic Iris dataset. It has been specifically generated to be compatible with the [Feast feature store](https://feast.dev/) and is intended for use in a hands-on tutorial.

Unlike the original static dataset, this version simulates the tracking of features for a few individual iris plants over a period of time, making it suitable for demonstrating real-world feature store concepts.

---

## The Problem with the Standard Iris Dataset

The standard Iris dataset is a simple table of 150 measurements. While excellent for basic classification tasks, it is unsuitable for demonstrating a feature store because it lacks:

1.  **An Entity**: There is no unique identifier for the object being measured (e.g., a specific plant ID). Feast requires an entity to associate features with.
2.  **Timestamps**: All data exists at a single, unknown point in time. Feast is built around time-series data to provide point-in-time correctness and prevent data leakage in training sets.

This dataset solves these issues by introducing an `iris_id` as the entity and an `event_timestamp` for each feature measurement.

---

## Dataset Schema

The data is stored in the `iris_data_adapted_for_feast.csv` file and has the following columns:

| Column Name         | Data Type | Description                                                                                                                                                             | Feast Role             |
| ------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `event_timestamp`   | Timestamp | The exact UTC timestamp when the measurement was recorded. This is crucial for historical lookups and point-in-time joins.                                               | **Timestamp Field** |
| `iris_id`           | Integer   | A unique identifier for each individual iris plant being tracked.                                                                                                         | **Entity Key** |
| `sepal_length`      | Float     | The length of the sepal in centimeters.                                                                                                                                 | Feature                |
| `sepal_width`       | Float     | The width of the sepal in centimeters.                                                                                                                                  | Feature                |
| `petal_length`      | Float     | The length of the petal in centimeters.                                                                                                                                 | Feature                |
| `petal_width`       | Float     | The width of the petal in centimeters.                                                                                                                                  | Feature                |
| `species`           | String    | The species of the iris plant (`setosa`, `versicolor`, or `virginica`). Can be used as a feature or a prediction target (label).                                          | Feature / Label        |
| `created_timestamp` | Timestamp | The UTC timestamp when the data row was created or ingested. Feast can use this to resolve data freshness.                                                                | **Created Timestamp** |

---

## How The Data Was Generated

This dataset was synthetically generated using a Python script:

1.  The base data comes from the `scikit-learn` Iris dataset.
2.  We simulated **3 unique iris plants** and assigned each an `iris_id` (1001, 1002, 1003).
3.  For each plant, we generated **15 days of sequential data**, creating a unique `event_timestamp` for each day.
4.  To simulate real-world variance, a small amount of random noise was added to the feature measurements (`sepal_length`, etc.) for each timestamp.
5.  The final DataFrame was saved in the efficient Parquet file format.

---