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
# model_candidates.py); raised again 2026-08-15 (4g->6g) since TUNING_PARALLELISM now
# means multiple tree ensembles can be mid-training on the driver JVM simultaneously,
# which increases peak memory pressure versus the earlier strictly-sequential fitting.
SPARK_DRIVER_MEMORY = _env("SPARK_DRIVER_MEMORY", "6g")
# Number of partitions Spark uses for shuffles (StringIndexer/CountVectorizer vocab
# building, groupBy, tree split-finding, etc.) and the target partition count for
# cv_train/validation/test after splitting. Spark's own default (200) is tuned for large
# multi-node clusters, not a single local[*] VM - on this project's ~29K-row cv_train
# slice, 200 tiny partitions add scheduling overhead without adding real parallelism.
# Set to roughly the VM's core count (see SparkSession.sparkContext.defaultParallelism,
# or `nproc` on the VM) so every individual model fit actually uses all available cores.
SPARK_SHUFFLE_PARTITIONS = _env_int("SPARK_SHUFFLE_PARTITIONS", 16)
# Number of (hyperparameter combination, CV fold) fits tune_models.cross_validate() runs
# concurrently, via background threads (see tune_models.py module docstring). Concurrency
# beyond 1 re-introduces the same class of risk as pyspark.ml.tuning.CrossValidator's own
# background-thread fitting, which caused a real ConnectionRefusedError in this project's
# VM Jupyter session before (PLAN.md §23 #13) - that's why tune_models.py replaced
# CrossValidator with a hand-rolled sequential loop in the first place. This setting
# reintroduces concurrency deliberately (to use the VM's 16 cores and cut wall-clock
# tuning time), but keep TUNING_PARALLELISM=1 as the known-safe fallback if that error
# recurs.
TUNING_PARALLELISM = _env_int("TUNING_PARALLELISM", 4)

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
# Employment is a multi-select field with ~9 real underlying statuses (see
# feature_config.py) - 10 comfortably covers all of them without an arbitrary cap.
TOP_EMPLOYMENT_STATUSES = _env_int("TOP_EMPLOYMENT_STATUSES", 10)
# §10.9 outlier decision (see data_cleaning.py docstring and PLAN.md §23): rows with a
# target above this are dropped as implausible (almost certainly data-entry errors, not
# real annual salaries - this dataset's max is $74,351,432) before log1p is applied to
# the rest. $1M is a generous upper bound for a tech-industry survey's self-reported
# annual salary (p99 in this dataset is ~$400K) - a domain-judgment call, not derived
# from the data itself, so it's configurable rather than hardcoded.
MAX_PLAUSIBLE_SALARY = _env_int("MAX_PLAUSIBLE_SALARY", 1_000_000)

# Producer / dashboard behavior
DATASET_EVENT_DELAY_SECONDS = _env_int("DATASET_EVENT_DELAY_SECONDS", 2)
DASHBOARD_PREDICTION_TIMEOUT_SECONDS = _env_int("DASHBOARD_PREDICTION_TIMEOUT_SECONDS", 30)
