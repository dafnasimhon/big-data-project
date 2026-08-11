import pytest
from pyspark.sql import SparkSession

from src.training.feature_pipeline import build_feature_pipeline

FEATURE_COLUMNS = [
    "Country",
    "Age",
    "EdLevel",
    "Employment",
    "RemoteWork",
    "DevType",
    "OrgSize",
    "Industry",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "YearsCodeProNumeric",
    "label",
    "log_label",
]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("test_feature_pipeline").getOrCreate()
    )
    yield session
    session.stop()


def _sample_df(spark):
    rows = [
        (
            "USA", "25-34 years old", "Bachelor's", "Employed, full-time", "Remote",
            "Developer", "20 to 99 employees", "Tech", "Python;SQL", "PostgreSQL", "AWS",
            5.0, 100000.0, 11.5129,
        ),
        (
            "Israel", "35-44 years old", "Master's", "Employed, full-time", "Hybrid",
            "Developer", "100 to 499 employees", "Tech", "Java;Python;SQL", "MySQL;PostgreSQL", "GCP",
            None, 90000.0, 11.4076,
        ),
        (
            "Germany", "45-54 years old", "PhD", "Employed, full-time", "In-person",
            "Manager", "10,000 or more employees", "Finance", "Unknown", "Unknown", "Unknown",
            15.0, 120000.0, 11.6952,
        ),
    ]
    return spark.createDataFrame(rows, FEATURE_COLUMNS)


def test_feature_pipeline_produces_features_column(spark):
    df = _sample_df(spark)
    model = build_feature_pipeline().fit(df)
    transformed = model.transform(df)

    assert "features" in transformed.columns
    assert transformed.filter(transformed.features.isNull()).count() == 0


def test_feature_pipeline_imputes_missing_years_code_pro(spark):
    df = _sample_df(spark)
    model = build_feature_pipeline().fit(df)
    transformed = model.transform(df)

    imputed_values = [row["YearsCodeProNumeric_imputed"] for row in transformed.collect()]
    assert None not in imputed_values


def test_feature_pipeline_handles_unseen_categories_at_transform_time(spark):
    train_df = _sample_df(spark)
    model = build_feature_pipeline().fit(train_df)

    unseen_row = spark.createDataFrame(
        [
            (
                "Freedonia", "18-24 years old", "Unknown", "Employed, full-time", "Remote",
                "Developer", "Unknown", "Unknown", "Rust;Zig", "Unknown", "Unknown",
                2.0, 60000.0, 11.0021,
            )
        ],
        FEATURE_COLUMNS,
    )

    # handleInvalid="keep" (StringIndexer/OneHotEncoder/VectorAssembler) must not raise on
    # categories never seen during fit — this is what makes the saved PipelineModel safe
    # to apply to live streaming requests with values outside the training vocabulary.
    transformed = model.transform(unseen_row)
    assert transformed.count() == 1
    assert transformed.first()["features"] is not None
