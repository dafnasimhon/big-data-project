"""Central configuration, loaded from environment variables / .env (PLAN.md rule 5)."""

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

# VM install locations (used by scripts/*.sh, not by Python directly)
KAFKA_HOME = _env("KAFKA_HOME", "/path/to/kafka")
SPARK_HOME = _env("SPARK_HOME", "/path/to/spark")

# Kafka
KAFKA_BOOTSTRAP_SERVERS = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_REQUEST_TOPIC = _env("KAFKA_REQUEST_TOPIC", "salary_requests")
KAFKA_PREDICTION_TOPIC = _env("KAFKA_PREDICTION_TOPIC", "salary_predictions")
KAFKA_DATASET_TOPIC = _env("KAFKA_DATASET_TOPIC", "developer_events")
KAFKA_DEAD_LETTER_TOPIC = _env("KAFKA_DEAD_LETTER_TOPIC", "salary_dead_letter")
KAFKA_ANALYTICS_TOPIC = _env("KAFKA_ANALYTICS_TOPIC", "salary_analytics")

# Spark
SPARK_MASTER_URL = _env("SPARK_MASTER_URL", "local[*]")
SPARK_APP_NAME = _env("SPARK_APP_NAME", "TechSalaryPrediction")
# Only takes effect when this process starts its own fresh JVM (e.g. spark-submit, a
# plain `python -m ...` run, or a brand-new Jupyter kernel with no Spark session created
# yet) - driver heap size is fixed at JVM startup and can't be changed on an
# already-running session. Raised from Spark's small default after a real
# OutOfMemoryError training RandomForestRegressor on the VM (2026-08-11, see
# model_candidates.py).
SPARK_DRIVER_MEMORY = _env("SPARK_DRIVER_MEMORY", "4g")

# Paths (relative to the repo root unless overridden)
DATASET_PATH = _env("DATASET_PATH", "./data/raw/survey_results_public.csv")
MODEL_PATH = _env("MODEL_PATH", "./models/best_salary_model")
MODEL_METADATA_PATH = _env("MODEL_METADATA_PATH", "./models/model_metadata.json")
MODEL_COMPARISON_PATH = _env("MODEL_COMPARISON_PATH", "./models/model_comparison.csv")
MODEL_METRICS_PATH = _env("MODEL_METRICS_PATH", "./models/model_metrics.json")

PREDICTION_CHECKPOINT_PATH = _env(
    "PREDICTION_CHECKPOINT_PATH", "./checkpoints/salary_predictions"
)
ANALYTICS_CHECKPOINT_PATH = _env(
    "ANALYTICS_CHECKPOINT_PATH", "./checkpoints/developer_events"
)

# Reproducibility / feature engineering
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
TOP_LANGUAGES = _env_int("TOP_LANGUAGES", 20)
TOP_DATABASES = _env_int("TOP_DATABASES", 15)
TOP_PLATFORMS = _env_int("TOP_PLATFORMS", 15)

# Producer / dashboard behavior
DATASET_EVENT_DELAY_SECONDS = _env_int("DATASET_EVENT_DELAY_SECONDS", 2)
DASHBOARD_PREDICTION_TIMEOUT_SECONDS = _env_int("DASHBOARD_PREDICTION_TIMEOUT_SECONDS", 30)
