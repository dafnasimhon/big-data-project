# Real-Time Tech Salary Prediction

A Spark + Kafka + Spark ML project that trains a salary-prediction model on the Stack
Overflow Developer Survey 2023, then serves real-time predictions over Kafka.

For the full project plan, architecture rationale, and a detailed history of issues found
and fixed during development, see [`PLAN.md`](PLAN.md).

## Architecture

- **Training** (`src/training/`) — cleans the raw survey data, builds a leakage-free
  Spark ML feature pipeline, tunes 4 candidate regression models via k-fold
  cross-validation, and saves the winner to `models/best_salary_model`.
- **Dataset producer** (`src/producers/dataset_producer.py`) — replays real historical
  survey rows onto the `developer_events` Kafka topic, simulating a live feed.
- **Prediction stream** (`src/streaming/prediction_stream.py`) — a Spark Structured
  Streaming job that reads salary-prediction requests from `salary_requests`, applies the
  trained model, and publishes results to `salary_predictions` (invalid requests go to
  `salary_dead_letter`).
- **Notebooks** (`notebooks/`) — this VM can't run Python/Spark scripts directly from a
  terminal, so every part of this project is run through a Jupyter notebook instead. See
  [Running the project](#running-the-project) below.

## Prerequisites

- Python 3.9+
- Apache Spark (version must match whatever's installed on the VM — check with
  `spark-submit --version`)
- Apache Kafka (this project was built against `kafka_2.13-3.2.1`)
- Jupyter (for running the notebooks)

## Setup

1. Clone the repo and `cd` into it.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (every notebook does this automatically on first run if
   it's missing) and adjust any paths/settings for your machine.
4. Make sure `data/raw/survey_results_public.csv` exists — extract it from `data.zip` if
   you have it (`unzip data.zip survey_results_public.csv -d data/raw/`), or provide the
   Stack Overflow Developer Survey 2023 CSV yourself.

## Running Kafka on the VM

```bash
alias cdk='cd /usr/local/kafka/kafka_2.13-3.2.1'
cdk
bin/zookeeper-server-start.sh config/zookeeper.properties &
bin/kafka-server-start.sh config/server.properties &
```

## Create the Kafka topics (one-time)

```bash
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic salary_requests

bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic salary_predictions

bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic developer_events

bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic salary_dead_letter
```

## Run Spark with Kafka support

```bash
pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2
```

Match the package version to whatever Spark version is actually installed
(`spark-submit --version`). The Kafka connector JAR can only be attached at process
launch — it can't be added to an already-running Spark session afterward, so this has to
happen before Jupyter (or any notebook's `SparkSession`) starts. If the VM is already
configured with this connector available by default, a plain `pyspark`/`jupyter notebook`
launch works too, without the `--packages` flag.

## Running the project

Everything runs through Jupyter notebooks, normally in this order:

1. **`notebooks/run_training_pipeline.ipynb`** — trains and tunes all 4 candidate models
   and saves the winner to `models/best_salary_model`. Run this first; everything else
   depends on a trained model existing.
2. **`notebooks/run_dataset_producer.ipynb`** — replays real historical survey rows onto
   `developer_events`.
3. **`notebooks/run_prediction_stream.ipynb`** — starts the real-time prediction stream
   (`salary_requests` → `salary_predictions`), publishes a test request and confirms a
   real prediction comes back, and includes a section that scores real historical events
   (from step 2) with the trained model and compares predicted vs. real salary.

## Testing

```bash
pytest tests/
```

## Project structure

```
config/settings.py          # central configuration, loaded from .env
src/
  common/                    # shared Spark session/schema/utility helpers
  training/                  # data cleaning, feature pipeline, model tuning/selection
  exploration/                # dataset exploration report (Phase 2)
  producers/                  # Kafka producer replaying the dataset
  streaming/                  # real-time Kafka prediction stream
notebooks/                    # how everything actually gets run on the VM
tests/                        # pytest suite
models/                       # trained model + metrics (generated, not committed)
data/                         # raw dataset + generated reports (mostly gitignored)
```
