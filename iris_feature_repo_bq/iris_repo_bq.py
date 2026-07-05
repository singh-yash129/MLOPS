
from datetime import timedelta
from feast import BigQuerySource, Entity, FeatureView, Field
from feast.types import Float32, String

iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    description="Unique identifier for each individual iris plant being tracked",
)

iris_source = BigQuerySource(
    name="iris_bq_source",
    table="<YOUR_GCP_PROJECT_ID>.iris_feast_dataset.iris_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

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
