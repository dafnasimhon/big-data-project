"""Central configuration, loaded from environment variables / .env (PLAN.md rule 5)."""

import os

from dotenv import load_dotenv

load_dotenv()

# This file lives at <project_root>/config/settings.py, so its own location - not
# whatever the current working directory happens to be - reliably anchors the project
# root. This matters more than it sounds: found on the VM (2026-08-15) that a relative
# DATASET_PATH resolved against the *Spark JVM's* working directory, not Python's -
# os.chdir() in a notebook only changes the Python process's cwd, but the JVM (a
# separate process reached via py4j) keeps whatever directory it was originally launched
# from, so the two can silently disagree. Anchoring every path setting to this absolute
# PROJECT_ROOT makes that mismatch impossible regardless of either process's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_path(name: str, default: str) -> str:
    """Like `_env`, but relative values are resolved to an absolute path under
    PROJECT_ROOT instead of being left relative to an unpredictable cwd."""
    value = os.environ.get(name, default)
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)

# VM install locations (used by scripts/*.sh, not by Python directly)
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

# Paths - resolved to absolute paths under PROJECT_ROOT (see _env_path above) so they're
# correct regardless of the calling process's (or the Spark JVM's) working directory.
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
# Same idea as MAX_PLAUSIBLE_SALARY, but for the low end - added 2026-08-20 after
# notebooks/run_prediction_stream.ipynb's real-vs-predicted comparison surfaced a real
# `developer_events` row with `ConvertedCompYearly = $14`, which `label > 0` (§10 step 7)
# lets through untouched since $14 is technically positive. $1,000/year is a generous
# floor - low enough that any conceivably real annual salary from even the lowest
# cost-of-living countries in this global survey stays in, while excluding obvious
# data-entry artifacts like $14. Also a domain-judgment call, not derived from the data.
MIN_PLAUSIBLE_SALARY = _env_int("MIN_PLAUSIBLE_SALARY", 1_000)

# Producer behavior
DATASET_EVENT_DELAY_SECONDS = _env_int("DATASET_EVENT_DELAY_SECONDS", 2)
