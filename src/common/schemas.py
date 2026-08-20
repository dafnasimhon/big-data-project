
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# Candidate input features shared by requests and dataset events (PLAN.md §2).
CANDIDATE_FEATURE_FIELDS = [
    StructField("Age", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("EdLevel", StringType(), True),
    StructField("RemoteWork", StringType(), True),
    StructField("Employment", StringType(), True),
    StructField("DevType", StringType(), True),
    StructField("YearsCodePro", DoubleType(), True),
    StructField("OrgSize", StringType(), True),
    StructField("LanguageHaveWorkedWith", StringType(), True),
    StructField("Industry", StringType(), True),
    StructField("DatabaseHaveWorkedWith", StringType(), True),
    StructField("PlatformHaveWorkedWith", StringType(), True),
]

# Topic: salary_requests — employee profile submitted by the dashboard (PLAN.md §8).
SALARY_REQUEST_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("event_time", StringType(), True),
        *CANDIDATE_FEATURE_FIELDS,
    ]
)

# Topic: salary_predictions — prediction result produced by Spark (PLAN.md §8).
SALARY_PREDICTION_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("prediction", DoubleType(), True),
        StructField("target_unit", StringType(), True),
        StructField("model_name", StringType(), True),
        StructField("model_version", StringType(), True),
        StructField("processed_at", StringType(), True),
        StructField("status", StringType(), True),
    ]
)

# Topic: developer_events — rows streamed gradually from the dataset (PLAN.md §14).
DEVELOPER_EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("event_time", StringType(), True),
        *CANDIDATE_FEATURE_FIELDS,
        StructField("ConvertedCompYearly", DoubleType(), True),
    ]
)

# Topic: salary_dead_letter — malformed/invalid events that couldn't be processed
# (PLAN.md §8, §15 step 3, §16 step 8).
DEAD_LETTER_SCHEMA = StructType(
    [
        StructField("source_topic", StringType(), True),
        StructField("raw_value", StringType(), True),
        StructField("error_reason", StringType(), True),
        StructField("received_at", StringType(), True),
    ]
)
