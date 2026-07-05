"""
Feast feature definitions for the IRIS pipeline (Task 2).

Defines:
- Entity: iris_id (uniquely identifies each iris plant being tracked)
- Data source: the time-aware iris dataset (parquet), with event_timestamp
  and created_timestamp columns
- Feature view: maps the iris measurement columns to the entity + source
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

# ---------------------------------------------------------------------------
# Entity: uniquely identifies each iris plant
# ---------------------------------------------------------------------------
iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    description="Unique identifier for each individual iris plant being tracked",
)

# ---------------------------------------------------------------------------
# Data source: points at the time-aware iris dataset.
# event_timestamp_column -> required by Feast for point-in-time correctness
# created_timestamp_column -> used to resolve freshness if duplicate rows exist
# ---------------------------------------------------------------------------
iris_source = FileSource(
    name="iris_source",
    path="data/iris_data_adapted_for_feast.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Feature view: maps the iris feature columns to the entity + source
# ---------------------------------------------------------------------------
iris_features_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=60),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    online=True,
    source=iris_source,
)