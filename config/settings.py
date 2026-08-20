"""Central configuration, loaded from environment variables / .env (PLAN.md rule 5)."""

import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_path(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)

KAFKA_HOME = _env("KAFKA_HOME", "/path/to/kafka")
SPARK_HOME = _env("SPARK_HOME", "/path/to/spark")

# Kafka
KAFKA_BOOTSTRAP_SERVERS = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_REQUEST_TOPIC = _env("KAFKA_REQUEST_TOPIC", "salary_requests")
KAFKA_PREDICTION_TOPIC = _env("KAFKA_PREDICTION_TOPIC", "salary_predictions")
KAFKA_DATASET_TOPIC = _env("KAFKA_DATASET_TOPIC", "developer_events")
KAFKA_DEAD_LETTER_TOPIC = _env("KAFKA_DEAD_LETTER_TOPIC", "salary_dead_letter")

# Spark
SPARK_MASTER_URL = _env("SPARK_MASTER_URL", "local[*]")
SPARK_APP_NAME = _env("SPARK_APP_NAME", "TechSalaryPrediction")
SPARK_DRIVER_MEMORY = _env("SPARK_DRIVER_MEMORY", "6g")
SPARK_SHUFFLE_PARTITIONS = _env_int("SPARK_SHUFFLE_PARTITIONS", 16)
TUNING_PARALLELISM = _env_int("TUNING_PARALLELISM", 4)
DATASET_PATH = _env_path("DATASET_PATH", "./data/raw/survey_results_public.csv")
MODEL_PATH = _env_path("MODEL_PATH", "./models/best_salary_model")
MODEL_METADATA_PATH = _env_path("MODEL_METADATA_PATH", "./models/model_metadata.json")
MODEL_COMPARISON_PATH = _env_path("MODEL_COMPARISON_PATH", "./models/model_comparison.csv")
MODEL_METRICS_PATH = _env_path("MODEL_METRICS_PATH", "./models/model_metrics.json")

PREDICTION_CHECKPOINT_PATH = _env_path(
    "PREDICTION_CHECKPOINT_PATH", "./checkpoints/salary_predictions"
)
DEAD_LETTER_CHECKPOINT_PATH = _env_path(
    "DEAD_LETTER_CHECKPOINT_PATH", "./checkpoints/salary_dead_letter"
)

# Reproducibility / feature engineering
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
TOP_LANGUAGES = _env_int("TOP_LANGUAGES", 20)
TOP_DATABASES = _env_int("TOP_DATABASES", 15)
TOP_PLATFORMS = _env_int("TOP_PLATFORMS", 15)

TOP_EMPLOYMENT_STATUSES = _env_int("TOP_EMPLOYMENT_STATUSES", 10)

MAX_PLAUSIBLE_SALARY = _env_int("MAX_PLAUSIBLE_SALARY", 1_000_000)

MIN_PLAUSIBLE_SALARY = _env_int("MIN_PLAUSIBLE_SALARY", 1_000)

# Producer behavior
DATASET_EVENT_DELAY_SECONDS = _env_int("DATASET_EVENT_DELAY_SECONDS", 2)
