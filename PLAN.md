# Real-Time Tech Salary Prediction — Project Plan

Big Data Analytics project using Apache Spark, Kafka and Spark ML.

**Core requirement:** train and compare several Spark ML regression models, select the best
model objectively using validation metrics, save it, and use it for real-time salary
prediction through Kafka and Spark Structured Streaming.

This file is the authoritative project specification, adapted from the supplied
implementation plan (`Claude_Code_Plan_Tech_Salary_Big_Data.pdf`) and the Kafka lab
(`Lab3 (3).pdf`). Repository/dataset inspection has been done once already (see
"Verified Dataset Facts" below); the rest of the plan is unchanged intent, restated as the
working spec for this repo.

---

## 1. Project Goal

Build an end-to-end Big Data system that predicts the expected annual salary of a
technology employee from personal and professional characteristics, demonstrating:

- Real-time data ingestion (Apache Kafka)
- Distributed stream processing (Spark Structured Streaming)
- Data cleaning/exploration/feature prep (Spark DataFrames)
- Reproducible preprocessing + training (Spark ML Pipelines)
- Comparison of several regression algorithms (not chosen in advance)
- Automatic, metric-driven best-model selection
- Real-time inference over Kafka
- An interactive dashboard (Streamlit) that sends requests and shows predictions
- Evaluation using RMSE, MAE and R²

## 2. Dataset and Research Question

**Dataset:** Stack Overflow Annual Developer Survey 2023, delivered as `data.zip`.

**Research question:** Can the salary of a technology employee be predicted accurately
from experience, education, geographic location, role and technical skills, while
processing large-scale data and prediction requests in real time?

**Candidate input features:**
`Age`, `Country`, `EdLevel`, `RemoteWork`, `Employment`, `DevType`, `YearsCodePro`,
`OrgSize`, `LanguageHaveWorkedWith`, `Industry`, `DatabaseHaveWorkedWith`,
`PlatformHaveWorkedWith`

Do **not** assume `CompTotal` is the correct target before inspecting the dataset. Check
`CompTotal`, `ConvertedCompYearly`, `Currency` and `CompFreq`. Prefer a normalized annual
salary target when the dataset supports it, and document the decision.

### Verified Dataset Facts (from `data.zip`, inspected 2026-08-04)

- Archive contains two files: `survey_results_public.csv` (~158 MB, the dataset) and
  `so_survey_2023.pdf` (the official survey codebook/documentation).
- **89,184 data rows** (+ 1 header row), 84 columns.
- All 11 candidate input features listed above **are present** in the raw CSV.
- Target-related columns present: `CompTotal`, `Currency`, `ConvertedCompYearly`.
- **`CompFreq` is NOT present** as a column in this CSV (unlike some other years of the
  survey). This means the raw pay frequency (yearly/monthly/weekly) used to compute the
  normalized figure is not independently available to us — we cannot re-derive an annual
  figure from `CompTotal` ourselves without it.
- `ConvertedCompYearly` is Stack Overflow's own pre-computed, currency-normalized,
  annualized compensation figure in USD. Given the missing `CompFreq`, **this is the
  strong candidate target column** (row 1 in the raw file is entirely `NA` — an example of
  a respondent who dropped out early, confirming heavy missingness must be handled).
- Row 1 (ResponseId=1) has `NA` across almost every field, confirming many partial/blank
  survey responses exist and will need explicit null-handling in cleaning.
- The full, formal target-selection report (row/column counts, missing-value rates per
  selected feature, duplicate detection, outlier quantiles, cardinality of categoricals,
  top languages/databases/platforms) is produced by the Phase 2 exploration script
  (`src/exploration/explore_dataset.py`) using Spark, per section 8 below — this is a
  planning-time spot check, not a substitute for that report.

### Final Target Decision (confirmed 2026-08-11 via notebook prototype)

- **Target column: `ConvertedCompYearly`** (USD, currency-normalized, pre-annualized by
  Stack Overflow). Confirmed by running the Phase 2 exploration against the real CSV.
- Of 89,184 rows, only **48,019 (53.8%) have a non-null, positive `ConvertedCompYearly`**;
  the other 41,165 rows are dropped during cleaning (no salary reported). This is the
  effective usable dataset size for training.
- 99th-percentile salary computed via `approxQuantile(..., relError=0.01)` came back equal
  to the raw max (\$74,351,432) — the approximation was too coarse to actually catch
  outliers, so percentile-based filtering did **nothing** in practice. The outlier/skew
  problem was instead addressed by training on `log1p(ConvertedCompYearly)` and reversing
  with `expm1` at inference time (decided in the model-improvement step, see §23).
  **This still needs a real outlier pass** (tighter `relError`, or hard bounds) — see
  Known Issues below.

## 3. Mandatory Development Rules

1. Inspect the repository and dataset before creating or changing files.
2. Do not delete or overwrite existing work without first reporting what will change.
3. Build the project incrementally and test each phase before continuing.
4. Use Spark for the main data-processing and machine-learning logic; do not replace it
   with Pandas.
5. Keep Kafka addresses, topic names, paths and model parameters in configuration files
   or environment variables.
6. Use explicit schemas for Kafka messages and structured data.
7. Prevent data leakage: all fitted preprocessing stages must be learned only from the
   training data.
8. Use logging, validation, checkpointing and graceful error handling.
9. Save evaluation reports, selected model metadata and reproducible random seeds.
10. Do not declare a model as best without a documented comparison on unseen
    validation/test data.

## 4. High-Level Architecture

```
A. MODEL DEVELOPMENT
Stack Overflow CSV
  -> Spark DataFrame
  -> Data exploration and cleaning
  -> Feature engineering pipeline
  -> Train / validation / test split
  -> Train several Spark ML regression models
  -> Hyperparameter tuning
  -> Compare RMSE, MAE and R²
  -> Select best model
  -> Final test evaluation
  -> Save best PipelineModel and metadata

B. REAL-TIME PREDICTION
Dashboard
  -> Kafka topic: salary_requests
  -> Spark Structured Streaming
  -> Load saved best model
  -> Generate prediction
  -> Kafka topic: salary_predictions
  -> Dashboard displays result

C. STREAMING ANALYTICS DEMONSTRATION
Stack Overflow CSV producer
  -> Kafka topic: developer_events
  -> Spark Structured Streaming
  -> Real-time aggregations
  -> Dashboard analytics
```

The first working version trains models **offline** and uses the selected saved model for
streaming inference. Do not retrain per event; periodic retraining is a later/advanced
feature only.

## 5. Technology Stack

| Component           | Technology                    |
|----------------------|-------------------------------|
| Language              | Python 3.11                  |
| Streaming broker       | Apache Kafka                 |
| Distributed engine     | Apache Spark / PySpark       |
| Streaming processing   | Spark Structured Streaming   |
| Machine learning       | Spark MLlib                  |
| Dashboard              | Streamlit                    |
| Infrastructure          | Target VM with Spark and Kafka pre-installed (no Docker) |
| Testing                | Pytest                       |

> **Environment note:** unlike the original plan's Docker-based setup, this project runs
> directly on a VM that already has Apache Spark and Apache Kafka installed. There is no
> `docker-compose.yml`, no containers, and no container health checks. Instead, Kafka and
> Spark are started the same way as in Lab3 (`bin/zookeeper-server-start.sh` /
> `bin/kafka-server-start.sh` from the Kafka install dir, `spark-submit` / a Spark
> standalone cluster for Spark), and every path (Kafka home, Spark home, dataset,
> checkpoints, models) is read from environment variables / `.env` so the project is not
> tied to one machine's exact install layout.

## 6. Kafka Working Pattern (from Lab3)

Lab3 established the hands-on Kafka mechanics this project should reuse rather than
reinvent:

- **Broker bring-up:** exactly as in Lab3 — ZooKeeper (or KRaft mode, whichever the VM's
  Kafka install is configured for) then the Kafka server, started from the Kafka home dir,
  e.g.:
  ```
  bin/zookeeper-server-start.sh config/zookeeper.properties &
  bin/kafka-server-start.sh config/server.properties &
  ```
  This project does **not** use Docker — Kafka and Spark are already installed on the
  target VM, so setup is a thin wrapper (`scripts/*.sh`) around the same `bin/*.sh`
  commands Lab3 uses, pointed at `$KAFKA_HOME`/`$SPARK_HOME` on that machine.
- **Topic lifecycle:** create/list via `kafka-topics.sh --create` / `--list`
  `--bootstrap-server`, mirrored here by `scripts/create_topics.sh` creating
  `salary_requests`, `salary_predictions`, `developer_events`, `salary_dead_letter`,
  `salary_analytics`.
- **Console producer/consumer** (`kafka-console-producer.sh`,
  `kafka-console-consumer.sh --from-beginning`) are useful smoke tests during development
  to confirm a topic is receiving/serving messages before wiring up Spark — use them
  ad hoc when debugging producers/streams in this project too.
- **Python producer via `confluent_kafka.Producer`**, `p.produce(topic, key=..., value=...)`
  — this is the same client pattern `src/producers/dataset_producer.py` and
  `src/producers/prediction_request_producer.py` will use, just with JSON payloads and an
  explicit schema instead of ad hoc strings.
- **Multiple producers at different intervals** (Lab3 §3: two producers at 1s vs 2.5s)
  maps to this project's configurable `DATASET_EVENT_DELAY_SECONDS` on the dataset
  producer.
- **Spark consuming Kafka** (Lab3 §4–6: DStream watching a directory, then real
  Structured Streaming reading a Kafka topic with periodic micro-batches, computing
  running aggregates like total lines / average rating / max ID) is the direct precedent
  for this project's `analytics_stream.py` (rolling aggregates: avg salary by
  country/role/experience) and `prediction_stream.py` (per-request inference). Use
  Structured Streaming (not DStreams) for both, with explicit schemas, watermarking, and
  checkpointing, as Lab3 §6.5–6.6 foreshadows.

## 7. Recommended Project Structure

```
salary-prediction-big-data/
|-- README.md
|-- PLAN.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|
|-- data/
|   |-- raw/survey_results_public.csv   (extracted from data.zip)
|   |-- samples/
|   `-- processed/
|
|-- models/
|   |-- best_salary_model/
|   |-- model_metrics.json
|   |-- model_comparison.csv
|   `-- model_metadata.json
|
|-- checkpoints/
|   |-- salary_predictions/
|   `-- developer_events/
|
|-- config/
|   `-- settings.py
|
|-- src/
|   |-- common/
|   |   |-- schemas.py
|   |   |-- spark_session.py
|   |   `-- logging_config.py
|   |
|   |-- exploration/
|   |   `-- explore_dataset.py
|   |
|   |-- training/
|   |   |-- data_loader.py
|   |   |-- data_cleaning.py
|   |   |-- feature_pipeline.py
|   |   |-- model_candidates.py
|   |   |-- tune_models.py
|   |   |-- select_best_model.py
|   |   |-- train_final_model.py
|   |   `-- evaluate_model.py
|   |
|   |-- producers/
|   |   |-- dataset_producer.py
|   |   `-- prediction_request_producer.py
|   |
|   |-- streaming/
|   |   |-- prediction_stream.py
|   |   `-- analytics_stream.py
|   |
|   `-- dashboard/
|       `-- app.py
|
|-- scripts/
|   |-- create_topics.sh
|   |-- train_and_select_model.sh
|   |-- start_prediction_stream.sh
|   |-- start_analytics_stream.sh
|   `-- start_dataset_producer.sh
|
`-- tests/
    |-- test_cleaning.py
    |-- test_schema.py
    |-- test_features.py
    |-- test_model_selection.py
    `-- test_prediction_flow.py
```

> **Current actual layout (2026-08-15).** The skeleton exists (`data/{raw,samples,
> processed}/`, `models/`, `checkpoints/`, `config/settings.py`, `scripts/`, `tests/` —
> see Phase 1 in §24), and Phases 2–5 have real, tested code in it, **confirmed working
> end-to-end against the real 89,184-row dataset on the actual VM, including outlier
> handling** (§23 #17): `src/common/{logging_config,spark_session,schemas,feature_config,
> spark_utils}.py`, `src/exploration/explore_dataset.py`, `src/training/{data_loader,
> data_cleaning,feature_pipeline,data_split,model_candidates,tune_models,
> select_best_model,train_final_model,evaluate_model}.py`, plus 31 passing pytest tests.
> `models/` has real content (`model_comparison.csv`, `model_metadata.json`,
> `model_metrics.json`, `best_salary_model/`) from the widened-grid, parallel-tuned run
> (§23 #20): selected model `LinearRegression`, final test R²=0.4684 — a legitimate,
> working, tuned result, not just passing code. Also added
> `notebooks/run_training_pipeline.ipynb`, a ready-to-run VM notebook with explicit Spark
> config verification. `src/streaming/prediction_stream.py` is also done and **confirmed
> against real Kafka on the VM** (§23 #21-23) — Kafka itself had to be stood up on the VM
> for the first time, with a genuinely important environment fix along the way (the
> Kafka connector JAR can only be added at PySpark process launch, not via `.config()`
> afterward; `notebooks/run_prediction_stream.ipynb` documents the working launch
> command and doubles as a self-contained test harness: starts the stream, publishes a
> test request via `confluent_kafka`, reads back the result). `src/{producers,
> dashboard}/` and `src/streaming/analytics_stream.py` are still not started.

## 8. Kafka Topics and Message Contracts

| Topic                | Purpose                                                      |
|-----------------------|----------------------------------------------------------------|
| `salary_requests`     | Employee profiles submitted by the dashboard.                 |
| `salary_predictions`  | Prediction results produced by Spark.                        |
| `developer_events`    | Rows streamed gradually from the dataset for the Big Data demo. |
| `salary_dead_letter`  | Malformed or invalid events that cannot be processed.         |
| `salary_analytics`    | Optional topic for real-time aggregate results.               |

**Example request:**
```json
{
  "request_id": "uuid",
  "event_time": "2026-08-04T12:00:00Z",
  "Age": "25-34 years old",
  "Country": "Israel",
  "EdLevel": "Bachelor's degree",
  "RemoteWork": "Hybrid",
  "Employment": "Employed, full-time",
  "DevType": "Data scientist or machine learning specialist",
  "YearsCodePro": 4,
  "OrgSize": "100 to 499 employees",
  "LanguageHaveWorkedWith": "Python;SQL;Java",
  "Industry": "Information Services, IT, Software Development",
  "DatabaseHaveWorkedWith": "PostgreSQL;MySQL",
  "PlatformHaveWorkedWith": "AWS;Azure"
}
```

**Example prediction response:**
```json
{
  "request_id": "uuid",
  "prediction": 125000.0,
  "target_unit": "annual salary",
  "model_name": "GBTRegressor",
  "model_version": "2026-08-04-01",
  "processed_at": "2026-08-04T12:00:03Z",
  "status": "success"
}
```

## 9. Dataset Exploration and Target Selection

Phase 2 must produce a Spark-based exploration report covering:

- Row/column/partition counts, schema and dtypes
- Missing values per selected column
- Duplicate rows and invalid numeric values
- Distributions of salary, country, experience, role
- Cardinality of categorical variables
- Most frequent languages, databases, platforms
- Availability/meaning of `CompTotal`, `ConvertedCompYearly`, `Currency`, `CompFreq`
  (confirmed absent in this dataset — see §2 above)
- Potential salary outliers and country/currency comparability

The report must explicitly state which salary column is the target, why, its unit, and
which rows are excluded. **Stop and report** any serious target/currency problem instead
of silently proceeding.

## 10. Data Cleaning

1. Select only required features and target columns.
2. Convert blank strings to null and remove duplicate rows.
3. Remove records with a missing or invalid target.
4. Cast the target and `YearsCodePro` to numeric values.
5. Convert `'Less than 1 year'` → `0.5` and `'More than 50 years'` → `51`.
6. Fill missing categorical values with `'Unknown'`.
7. Filter non-positive salaries.
8. Investigate extreme outliers using approximate quantiles.
9. Choose either controlled outlier filtering or `log1p` transformation of the target,
   and document the choice.
   > **Revised to use both together (2026-08-11).** The original log1p-only choice
   > wasn't sufficient — the first real-data training run showed every model's RMSE
   > dominated by extreme salary values (max $74,351,432 in this dataset) even with
   > log1p applied (§23 #15). `src/training/data_cleaning.py` now drops rows with
   > `label > MAX_PLAUSIBLE_SALARY` (default $1,000,000, `config/settings.py`) as
   > implausible — almost certainly data-entry errors, not real salaries — *before*
   > applying log1p to the rest. $1M is a documented domain-judgment call (this
   > dataset's p99 is ~$400K), not derived from the data, so it's configurable. Code
   > change is unit-tested (`tests/test_cleaning.py::
   > test_clean_dataset_drops_implausibly_high_salaries`); **re-running Phase 4/5
   > against the real dataset to confirm it actually improves RMSE/R² is the
   > next step** — see §23 #16.
10. Save row counts before and after every major cleaning step.

## 11. Feature Engineering

**Single-value categoricals** (`Age`, `Country`, `EdLevel`, `RemoteWork`, `Employment`,
`DevType`, `OrgSize`, `Industry`): `StringIndexer(handleInvalid='keep')` → `OneHotEncoder`.

**Numeric:** `YearsCodePro` as numeric; `Imputer` if needed; `StandardScaler` optional,
but the same feature vector must be available to every candidate model.

**Multi-value categoricals** (`LanguageHaveWorkedWith`, `DatabaseHaveWorkedWith`,
`PlatformHaveWorkedWith`, semicolon-separated) — do **not** index the full combined
string as one category:
1. Split each field by `;`.
2. Determine most frequent values from **training data only**.
3. Keep a configurable top-N list per field.
4. Create binary indicator features for those values.
5. Optionally add an `'other'` indicator.
6. Save the exact vocabulary in model metadata for identical streaming transformations.

## 12. Model Training, Comparison and Automatic Selection

Linear Regression is only a baseline — it must **not** be preselected as the final model.

**Required candidate models:**

| Model                     | Role                        | Initial params to tune                              |
|-----------------------------|------------------------------|--------------------------------------------------------|
| `LinearRegression`           | Interpretable baseline       | `regParam`, `elasticNetParam`, `maxIter`               |
| `DecisionTreeRegressor`       | Non-linear tree baseline     | `maxDepth`, `minInstancesPerNode`, `maxBins`            |
| `RandomForestRegressor`       | Robust ensemble model        | `numTrees`, `maxDepth`, `featureSubsetStrategy`, `maxBins` |
| `GBTRegressor`                | Boosted non-linear model     | `maxIter`, `maxDepth`, `stepSize`, `maxBins`             |

Optional models only if cleanly supported in Spark ML and non-blocking to the required
comparison.

**Data splitting strategy:**
- Hold out 20% as an untouched final test set.
- Remaining 80% for train/validation.
- Fixed seed `42`.
- `CrossValidator` or `TrainValidationSplit` on the training portion.
- Indexers/encoders/vocabularies/imputers fitted **inside each candidate Pipeline** to
  prevent leakage.

**Model-selection rule:**
1. Primary: lowest validation RMSE.
2. Tie-breaker 1: lowest validation MAE.
3. Tie-breaker 2: highest validation R².
4. Record training time/complexity, but don't override a clearly better predictive model
   without documenting why.
5. Evaluate the winner once on the untouched test set.
6. Save the full winning `PipelineModel`, not just the estimator.

**Required comparison output** → `models/model_comparison.csv` (Model, Best Parameters,
Validation RMSE, Validation MAE, Validation R², Training Time) and
`models/model_metadata.json`:
```json
{
  "selected_model": "GBTRegressor",
  "selection_metric": "validation_rmse",
  "best_parameters": {},
  "validation_metrics": {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
  "test_metrics": {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
  "target_column": "ConvertedCompYearly",
  "target_transformation": "log1p_or_none",
  "feature_version": "v1",
  "trained_at": ""
}
```

## 13. Baseline and Evaluation Requirements

- Also compute a naive mean-prediction baseline; compare every trained model against it.
- RMSE = primary selection metric; MAE = interpretable average error; R² = variance
  explained.
- Report all metrics in the **original salary scale**, even when trained on
  `log1p(target)`.
- Reject any pipeline producing NaN/infinite/negative final predictions without
  controlled post-processing.
- Store final test metrics in `models/model_metrics.json`.

## 14. Dataset Kafka Producer

- Read the CSV gradually; select relevant fields; convert each row to JSON.
- Add `event_id`, `event_time`.
- Publish to `developer_events` with configurable delay.
- Graceful shutdown, retries, delivery logging.
- CLI for file, topic, delay (mirrors Lab3 §3's configurable-interval producers).

## 15. Spark Real-Time Analytics Stream

- Read `developer_events` with an explicit schema.
- Parse/validate JSON without collecting the full stream to the driver.
- Malformed records → `salary_dead_letter`.
- Compute event counts, avg salary by country/role/experience range, common technologies.
- Event-time windows, watermarking, checkpointing where appropriate.
- Sink to Kafka, Parquet, or another documented structured sink.

## 16. Spark Real-Time Prediction Stream

1. Start Spark with the required Kafka connector.
2. Load `models/best_salary_model` and `model_metadata.json`.
3. Read `salary_requests` with an explicit schema.
4. Validate required fields; preserve `request_id`.
5. Apply the exact fitted feature pipeline + winning regression model.
6. Reverse any target transformation (e.g. `expm1`).
7. Build a prediction response with model name/version.
8. Publish valid results to `salary_predictions`; invalid → `salary_dead_letter`.
9. Use a checkpoint directory and graceful shutdown.

## 17. Streamlit Dashboard

**Personal prediction page:**
- Inputs for all selected features; generate a UUID per request.
- Publish profile to `salary_requests`; await matching `salary_predictions` by
  `request_id`.
- Display predicted salary, model name, processing time, timestamp.
- Clear timeout/error messaging.
- Disclaimer: result is a survey-based estimate, may differ from actual salaries.

**Descriptive analytics page:**
- Salary by country, by years of professional experience, by developer role, by
  language/technology.
- Salary distribution.
- Count of Kafka events processed.

## 18. VM Execution Environment (replaces Docker Compose)

The original plan's Docker Compose section is **not used** in this project. Spark and
Kafka are already installed on the target VM, so distributed execution is achieved
without containers:

- Kafka broker(s) started via the VM's existing Kafka install (`$KAFKA_HOME/bin/*.sh`,
  ZooKeeper or KRaft per however that install is configured) — same commands as Lab3.
- Spark: use whatever the VM already provides. If it has a standalone cluster running
  (one master + workers), point `SPARK_MASTER_URL` at `spark://<host>:7077` and submit
  jobs with `spark-submit`. If only a single-node Spark install is available, run with
  `local[*]` — both modes must work, selected purely via `SPARK_MASTER_URL`.
- No container health checks — instead, each `scripts/*.sh` wrapper checks the relevant
  process/port before proceeding (e.g. topic creation waits for the broker to accept
  connections) and fails loudly with a clear message rather than hanging silently.
- No Docker volumes — `data/`, `models/`, `checkpoints/` are plain local directories
  under the repo (or paths pointed to elsewhere on the VM via `.env`), created by Phase 1
  setup and left out of git via `.gitignore`.
- All Kafka/Spark/model/path settings still come from `config/settings.py` + `.env`
  (rule 5) exactly as before — only the values change (local paths instead of `/app/...`
  container paths, `localhost:9092` instead of `kafka:9092` unless told otherwise).

## 19. Configuration

```
KAFKA_HOME=/path/to/kafka        # set to the VM's actual Kafka install dir
SPARK_HOME=/path/to/spark        # set to the VM's actual Spark install dir

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_REQUEST_TOPIC=salary_requests
KAFKA_PREDICTION_TOPIC=salary_predictions
KAFKA_DATASET_TOPIC=developer_events
KAFKA_DEAD_LETTER_TOPIC=salary_dead_letter
KAFKA_ANALYTICS_TOPIC=salary_analytics

SPARK_MASTER_URL=local[*]        # or spark://<host>:7077 if a standalone cluster is up
SPARK_APP_NAME=TechSalaryPrediction

DATASET_PATH=./data/raw/survey_results_public.csv
MODEL_PATH=./models/best_salary_model
MODEL_METADATA_PATH=./models/model_metadata.json
MODEL_COMPARISON_PATH=./models/model_comparison.csv

PREDICTION_CHECKPOINT_PATH=./checkpoints/salary_predictions
ANALYTICS_CHECKPOINT_PATH=./checkpoints/developer_events

RANDOM_SEED=42
TOP_LANGUAGES=20
TOP_DATABASES=15
TOP_PLATFORMS=15
DATASET_EVENT_DELAY_SECONDS=2
DASHBOARD_PREDICTION_TIMEOUT_SECONDS=30
```

## 20. Testing Requirements

- `YearsCodePro` conversion.
- Null/invalid salary handling.
- Kafka JSON schema validation.
- Multi-value feature parsing.
- No leakage between training and test processing.
- Model-comparison sorting and tie-break rules.
- Save/reload of the winning `PipelineModel`.
- Preservation of `request_id`.
- Numeric, non-negative prediction output.
- End-to-end Kafka request-to-response integration test using a small sample dataset.

## 21. Implementation Phases

| Phase | Work                                            | Completion check                             |
|-------|---------------------------------------------------|-------------------------------------------------|
| 1     | Repository, configuration, Spark and Kafka setup on the VM | Kafka broker and Spark are reachable and topics exist |
| 2     | Dataset exploration and target decision            | Target report and data-quality report saved   |
| 3     | Cleaning and reusable feature pipeline             | Transformations pass unit tests               |
| 4     | Train and tune candidate models                    | Comparison table produced                     |
| 5     | Select winner and final test evaluation            | Best PipelineModel and metadata saved         |
| 6     | Kafka dataset producer                             | Events appear in `developer_events`           |
| 7     | Prediction and analytics streams                   | Spark consumes and publishes valid results    |
| 8     | Dashboard                                          | End-to-end request returns a prediction       |
| 9     | Testing and documentation                          | README and automated tests complete           |

## 22. Acceptance Criteria

1. Kafka starts successfully and all required topics exist.
2. Spark runs on the VM (standalone cluster if available, otherwise `local[*]`).
3. The dataset is loaded and explored using Spark.
4. The target salary column is selected and justified.
5. Cleaning and feature engineering are implemented as reproducible Spark Pipelines.
6. At least Linear Regression, Decision Tree, Random Forest and GBT are trained and tuned.
7. A model-comparison file contains validation RMSE, MAE, R² and best parameters.
8. The winning model is selected automatically using the documented rule.
9. The winning model is evaluated once on an untouched test set.
10. The complete winning PipelineModel is saved and reloadable.
11. A producer streams dataset records to Kafka.
12. The dashboard publishes prediction requests to Kafka.
13. Spark consumes requests and publishes matching predictions.
14. The dashboard displays the result by `request_id`.
15. Invalid events are handled through a dead-letter path.
16. Streaming jobs use checkpointing.
17. The complete project is documented and reproducible on the VM via `scripts/*.sh`
    (no Docker required).

## 23. Status / Next Step

**Outlier handling is done and confirmed working (2026-08-15) — R² went from ≈0 to
0.46.** §10.9 originally chose log1p-only (no hard filtering), based on the notebook
prototype's results. The first successful real-data run of the full Phase 4/5 pipeline
(2026-08-11, §23 #15) showed that wasn't sufficient: every model's R² was ≈0 and RMSE
(~700K) was wildly out of proportion to MAE (~43-49K) — a handful of extreme salary
values (max $74,351,432) dominating the metric. Fixed by dropping rows above
`MAX_PLAUSIBLE_SALARY` ($1M) before `log1p` (§23 #16), and **confirmed decisively on the
VM (§23 #17): RMSE dropped ~11-12x (700K→59K), R² rose from 0.0025 to 0.4623 (validation)
/ 0.4585 (final test)**, from dropping just ~0.13% of rows. Phase 4/5 is now in solid
shape both procedurally and substantively.

**Widened the tuning grids with real parallelism — done and confirmed better
(2026-08-15).** §23 #14 had cut grids hard (1 tuned hyperparameter/2 values per model,
`NUM_FOLDS=2`) for speed/memory reasons. Reverted that cut properly rather than just
restoring the old numbers: `NUM_FOLDS` back to 3, each model now tunes 2 hyperparameters
over ~6 combinations (76 total pipeline fits, vs. 20 before), and
`tune_models.cross_validate()` runs `TUNING_PARALLELISM` (default 4) of those fits
*concurrently* via background threads on the VM's 16 cores — see §23 #18-20 for the full
reasoning (including the deliberate `ConnectionRefusedError` risk trade-off) and a real
path-resolution bug (§23 #19) found and fixed along the way. **Confirmed on the VM (§23
#20): 76 fits completed in ~315s — about the same wall-clock time the narrower 20-fit run
took, i.e. ~4x the search for no extra time — and validation R² improved 0.4623→0.4750,
final test R² 0.4585→0.4684.** Phase 4/5 is now done, tested, and tuned about as far as
this feature set is likely to go without new features or a different modeling approach.

**Phase 7 prediction stream ported and confirmed against real Kafka (2026-08-15).**
`src/streaming/prediction_stream.py` reads `salary_requests`, applies the saved model,
reverses `log1p`, publishes to `salary_predictions`, and routes invalid requests to
`salary_dead_letter` — fixing several gaps in the notebook prototype (§23 #21-23 for the
full story, including standing up Kafka on the VM for the first time and a genuinely
important environment fix: the Kafka connector JAR can only be added at PySpark process
launch, not via `.config()` afterward — `notebooks/run_prediction_stream.ipynb`
documents the working launch command). **Confirmed end-to-end**: a real request got a
real prediction (`$82,046.35`), an invalid one correctly landed on the dead-letter topic.

Next options, roughly in likely order of value:
- `src/streaming/analytics_stream.py` (§15/Phase 7.2) — the other half of Phase 7,
  not started (windowed aggregates over `developer_events`).
- `src/producers/dataset_producer.py` (Phase 6) — needed to actually feed
  `developer_events` for the analytics stream to have something to consume.
- `scripts/start_prediction_stream.sh` / `scripts/train_and_select_model.sh` (§24
  Phases 5.4/7.3) — thin shell wrappers around already-working Python entry points.
- Phase 8 (dashboard), Phase 9 (tests/docs) — not started at all yet.

Completed so far (investigation only, per the plan's own "begin with investigation and
planning only" instruction):
- Confirmed `data.zip` contains `survey_results_public.csv` (89,184 rows, 84 columns) and
  the official `so_survey_2023.pdf` codebook.
- Confirmed all 11 candidate input features exist in the raw CSV.
- Confirmed `CompTotal`, `Currency`, `ConvertedCompYearly` exist; `CompFreq` does **not**
  exist in this file, which points toward `ConvertedCompYearly` as the target (to be
  finalized by the full Phase 2 exploration script, not by this spot check).
- Captured the Kafka working pattern from Lab3 to reuse for topic setup, producers, and
  Spark Structured Streaming consumption (§6 above).

### Notebook prototype added (2026-08-11)

A separate folder, `big_data_project/` (containing `notebooks/` and `data/`, opened as an
additional workspace folder alongside this repo, not yet merged into it), was created and
used to spike Phases 2–5 and part of Phase 7 end-to-end in Jupyter notebooks against a
live PySpark shell, ahead of building the planned `src/` module structure. This was
investigation/prototyping work, not the final implementation — see "Known issues" below
before treating any of it as final.

**What it contains and validates:**
- `notebooks/01_Data_Exploration.ipynb` — Spark-based exploration matching §9: schema,
  per-column missing-value counts, salary summary stats/percentiles, groupbys by country/
  age/EdLevel/RemoteWork/Employment/DevType/OrgSize/Industry, sample of the multi-value
  skill columns. Confirms the target decision in §2 above. Report saved to
  `output/missing_values_summary` (CSV), not yet `data/processed/` as §7 specifies.
- `notebooks/02_Data_Preprocessing.ipynb` — implements most of §10: casts `label` and
  `YearsCodePro` (`'Less than 1 year'` → 0.5, numeric → numeric), drops the 41,165 rows
  with no usable salary, fills missing categoricals/skill columns with `'Unknown'`,
  imputes missing `YearsCodePro` with the training median (8.0). Saves the cleaned
  48,019-row dataset as Parquet to `data/clean_salary_data/`. Does **not** implement outlier
  filtering (see §2 note) or before/after row-count logging at every step (§10.10).
- `notebooks/03_Model_Training.ipynb` — builds the feature pipeline per §11 (single-value
  categoricals: `StringIndexer(handleInvalid='keep')` → `OneHotEncoder` → `VectorAssembler`
  with `YearsCodeProNumeric`), trains all 4 required models (LinearRegression,
  DecisionTreeRegressor, RandomForestRegressor, GBTRegressor) with one hardcoded
  parameter set each (no tuning yet), evaluates with MAE/RMSE/R² via
  `RegressionEvaluator`, selects Random Forest as the winner by lowest RMSE, saves the
  full `PipelineModel` to `models/best_salary_model/`, and verifies it reloads and predicts
  correctly. Initial result: **RMSE 256,378 / MAE 49,871 / R² -0.096** (worse than a naive
  mean predictor).
- `notebooks/03B_Model_Improvement.ipynb` — adds `LanguageCount`/`DatabaseCount`/
  `PlatformCount` (list length of each `;`-separated skill field) as numeric features
  instead of the top-N binary indicators §11 specifies, retrains Random Forest against
  `log1p(label)` (reversed with `expm1` at inference), and overwrites
  `models/best_salary_model/` since it scored better. Result: **RMSE 257,705 / MAE 46,895 /
  R² 0.012** — still barely above baseline.
- `notebooks/03C_Model_KPIs.ipynb` — computes a median-salary baseline (§13) and confirms
  the trained model beats it on both MAE and RMSE; adds median/P90 absolute error,
  \"percent of predictions within X% of actual\" bands, and per-country/per-experience-band
  breakdowns. Saves KPI CSVs under `output/`.
- `notebooks/04_Spark_Streaming_Prediction.ipynb` — proves out the riskiest architectural
  piece: real Spark Structured Streaming reading `salary_requests` from Kafka
  (`localhost:9092`), parsing with an explicit `StructType` schema, applying the saved
  `PipelineModel`, reversing `log1p` via `expm1`, and writing predictions to both an
  in-memory table (for inspection) and the `salary_predictions` Kafka topic. Ran
  successfully against manually-published test requests. Uses a Spark-generated temp
  checkpoint, not the plan's `checkpoints/` directory; no dead-letter handling; no schema
  validation of required fields.

**Known issues to fix before this is "final" (not optional polish):**
1. ~~Test-set leakage across notebooks~~ **Resolved (2026-08-11).**
   `src/training/data_split.py` now carves the cleaned dataset into three disjoint
   slices: `cv_train` (60%, all `CrossValidator` ever sees), `validation` (20%, used
   exactly once per candidate model to compute the real-scale metrics
   `select_best_model.py` compares), and `test` (20%, referenced only inside
   `train_final_model.py`, exactly once, after the winner is already chosen). See §24
   Phase 4/5.
2. ~~No real hyperparameter tuning~~ **Resolved (2026-08-11).**
   `src/training/model_candidates.py` + `tune_models.py` now run a real
   `CrossValidator` (3 folds) per model over a `ParamGridBuilder` grid for each of the 4
   required models, tuned against log-space RMSE. Grids are deliberately small (~4
   combinations each) — documented in `model_candidates.py` as appropriate for a
   dev machine; widen them for the full run on the VM.
3. ~~Multi-value skill features simplified to counts~~ **Resolved (2026-08-11).**
   `src/training/feature_pipeline.py` now implements §11 as written: `RegexTokenizer` +
   `CountVectorizer(binary=True)` per skill column, building top-N (`TOP_LANGUAGES`/
   `TOP_DATABASES`/`TOP_PLATFORMS`) binary indicators fit only on the training fold —
   see §24 Phase 3.
4. **Model quality is weak** — R² of 0.012 (notebook prototype) is barely above a flat
   baseline. **Still open** — the ported Phase 4/5 code (§24) hasn't been run against the
   real 89,184-row dataset yet (blocked on 1.3, the CSV isn't in `data/raw/` in this repo),
   so there's no real-scale number for the ported pipeline yet. Worth revisiting
   `Country`/`DevType` one-hot cardinality and whether the now-available per-skill
   indicators (#3) help more than the notebook's raw counts did, once it runs for real.
5. ~~Output artifacts don't match §12's required paths/schema~~ **Resolved (2026-08-11).**
   `src/training/evaluate_model.py` writes `models/model_comparison.csv`,
   `models/model_metadata.json`, `models/model_metrics.json` in exactly the documented
   schema (verified with a dummy-data smoke test, see §24 Phase 5), and saves the winning
   `PipelineModel` to `models/best_salary_model/`. Not yet exercised against real data
   (same blocker as #4).
6. ~~Not yet ported to `src/`~~ **Further resolved (2026-08-11).** Phase 4/5
   (training/tuning/selection) now also lives in `src/training/`:
   `data_split.py`, `model_candidates.py`, `tune_models.py`, `select_best_model.py`,
   `train_final_model.py`, `evaluate_model.py`, with 24/24 pytest tests passing locally
   (added `test_model_candidates.py`, `test_select_best_model.py`,
   `test_training_pipeline.py` this round). Phase 7 streaming logic is still **not
   ported** — that remains open.
7. **Found and fixed during porting (2026-08-11).** The prototype's bare `.cast("double")` on
   non-numeric strings (e.g. `"NA"`) relies on Spark's ANSI SQL mode being off (the VM's
   Spark 3.3.0 default) to silently return null — under ANSI mode (the default in newer
   Spark) the same cast *raises an exception* instead. `src/common/spark_utils.py:
   safe_cast_double()` fixes this with a regex-validated cast that behaves the same
   either way; `data_cleaning.py` and `explore_dataset.py` both use it now. This was
   only caught because the ported code was actually tested against real PySpark rather
   than assumed correct from the notebook's observed behavior.
8. **Found and fixed during porting (2026-08-11).** `pyspark.ml.param.Params.set()`
   mutates the object in place and returns `None`, not `self` — `train_final_model.py`'s
   first draft did `regressor = regressor.set(...)` in a loop over `best_params`, which
   set `regressor` to `None` after the first parameter and crashed on the second. Fixed
   by calling `.set(...)` without reassigning. Also only caught by actually running the
   test suite against real PySpark, not by code review alone.
9. **Unverified on the VM's exact Spark version.** Everything in §24 Phases 2–5 has been
   tested locally against PySpark 4.2.0, not the VM's pinned Spark 3.3.0. The APIs used
   (`CrossValidator`, `ParamGridBuilder`, `CountVectorizer`, `RegexTokenizer`, `Imputer`)
   are stable across that range, but two real bugs (#7, #8) were only found by actually
   running code, not by inspection — treat this as a real risk, not a formality, until
   it's run there.
10. **Found and fixed running on the real VM (2026-08-11).** Eight files used
    `str | None`-style union type hints (PEP 604, Python 3.10+ only) — these crash with
    `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` at import time
    on the VM's older Python, since default-argument annotations are evaluated eagerly.
    Fixed by adding `from __future__ import annotations` to
    `src/common/spark_session.py`, `src/training/{data_loader,data_split,
    train_final_model,evaluate_model,tune_models,select_best_model,
    model_candidates}.py` — this defers annotation evaluation so the exact Python
    version stops mattering. Confirms Known Issue #9 wasn't just caution: running
    on the real target environment immediately surfaced a bug local testing couldn't.
11. ~~Full run against the real dataset blocked on this Windows dev machine~~
    **Superseded (2026-08-11).** Confirmed unrelated to the target VM — the user ran
    `explore_dataset.build_report()` directly on the actual VM (`~/project`, real Spark
    install) against the full 89,184-row file, in a Jupyter notebook (`get_spark_session`
    + `build_report` called directly rather than via `main()`, so the JSON file write
    didn't happen, but the in-memory report did — no winutils-equivalent issue hit on
    Linux, as expected). This produced the report seen in #12 below, which is what
    surfaced that bug. Local Windows runs remain blocked on `winutils.exe` for anything
    that writes files, but that's no longer the only way to validate against real data.
12. **Found and fixed running on the real VM (2026-08-11) — the important one.** The
    exploration report claimed `rows_with_target_present: 89184,
    rows_with_target_missing: 0` — i.e. 100% of rows have a usable salary, which
    contradicts the already-established fact that only 48,019/89,184 (53.8%) do (§2). Root
    cause: **this dataset encodes missing values as the literal text string `"NA"`**, not
    SQL null or a blank string (Spark's CSV reader's `nullValue` option defaults to `""`,
    not `"NA"`, so it's read as ordinary text). Every missing-value check in the codebase
    that only tested `isNull()`/blank-string — `explore_dataset.py`'s
    `missing_value_summary()` and `salary_present` check, and, more seriously,
    `data_cleaning.py` step 2's blank-to-null pass — missed it entirely. For
    `data_cleaning.py` specifically, this meant step 6's `fillna("Unknown", ...)` never
    actually caught `"NA"` values in categorical/skill columns (`fillna` only touches true
    nulls), so `"NA"` was sitting in the training data as its own bogus category instead
    of being merged into `"Unknown"` as §10.6 requires. (The target column itself was
    *not* affected — `safe_cast_double`'s regex validation already rejected `"NA"` as
    non-numeric, which is why the 48,019-row cleaned dataset figure was always correct.)
    Fixed with a new shared helper, `src/common/spark_utils.py: is_missing_text()`, used
    by both `data_cleaning.py` step 2 and `explore_dataset.py`'s missing-value checks.
    New regression tests added: `tests/test_spark_utils.py`, `tests/test_explore_dataset.py`,
    and `tests/test_cleaning.py::test_clean_dataset_fills_na_string_categoricals_with_unknown`
    — the existing categorical-fill test used Python `None` instead of the real `"NA"`
    sentinel, which is exactly why it didn't catch this. 30/30 tests passing locally
    after the fix. **Confirmed fixed on the VM (2026-08-11):** re-run against the real
    89,184-row file now reports `rows_with_target_present: 48019,
    rows_with_target_missing: 41165` — exactly matching the known 53.8% figure from §2.
    A related follow-up was found in the same report: `top_values()` (used for
    top languages/databases/platforms) was also counting `"NA"` as if it were a real
    choice — it ranked 2nd in `top_platforms` and 6th in `top_databases` on the real data.
    Fixed the same way (reusing `is_missing_text()`), verified with a new test
    (`test_top_values_excludes_na_sentinel`), not yet re-confirmed against real data (low
    risk — same fix pattern, already proven correct once this session).
13. **Found and fixed running on the real VM (2026-08-11).** `run_training_pipeline()`
    crashed with `Py4JJavaError: ... ConnectionRefusedError: [Errno 111] Connection
    refused` inside `CrossValidator._fit()`, reproduced twice (including mid-
    `RandomForestRegressor`). Root cause: `CrossValidator` fits each fold from a
    background thread (even at its default `parallelism=1`), and that thread needs its
    own fresh py4j socket connection back to the JVM gateway — unreliable under load in
    this VM's Jupyter kernel environment, even though every same-thread Spark call worked
    fine throughout. Fixed by replacing `CrossValidator` with a hand-rolled sequential
    k-fold loop (`src/training/tune_models.py: cross_validate()`) that does identical
    fold-splitting/fit/evaluate/refit-on-all-data work entirely on the calling thread. No
    performance cost — `parallelism=1` meant `CrossValidator` never actually ran folds
    concurrently anyway.
14. **Found and fixed running on the real VM (2026-08-11).** After #13's fix,
    `RandomForestRegressor` then failed with a genuine
    `java.lang.OutOfMemoryError: Java heap space` inside `RandomForests.findBestSplits`
    (`collectAsMap`). Two contributing causes, both fixed:
    - `Employment` (originally in `SINGLE_VALUE_CATEGORICAL_COLUMNS`) is actually a
      `;`-separated multi-select field (`"Employed, full-time;Student, part-time"`), not
      single-valued — one-hot-encoding it treated every unique *combination* as its own
      category, inflating it to ~107 dimensions in the real data (vs. ~9 real underlying
      statuses). Moved to the same `RegexTokenizer` + `CountVectorizer` treatment as the
      skill columns (now `MULTI_VALUE_COLUMNS` in `feature_config.py`) — both a
      correctness fix (each status is now its own meaningful boolean signal instead of an
      arbitrary combination id) and a large cut in feature-vector width.
    - The tuning grids and CV fold count were reduced (`NUM_FOLDS` 3→2; each model now
      tunes one hyperparameter over 2 values instead of two hyperparameters/~4
      combinations) — 5 pipeline fits per model instead of 13, 20 total instead of 52.
      Also cut total wall-clock time substantially, which was a separate complaint (a
      single `LinearRegression` tuning pass was taking ~86s even before the OOM).
    Verified: 24/24 tests passing locally, and **the user then successfully ran the full
    `run_training_pipeline()` against the real 89,184-row dataset in the VM notebook —
    see #15.**
15. **First successful full run against real data, VM notebook (2026-08-11).** Split
    sizes: cv_train 28,696 / validation 9,685 / test 9,635 (of 48,016 cleaned rows — 3
    fewer than the 48,019 in §2, from de-duplication now happening during cleaning, which
    earlier ad hoc checks didn't include). All 4 models tuned successfully:

    | Model | Validation RMSE | Validation MAE | Validation R² | Best params |
    |---|---|---|---|---|
    | LinearRegression (selected) | 700,079 | 43,702 | 0.0025 | regParam=0.1 |
    | DecisionTreeRegressor | 700,162 | 49,037 | 0.0022 | maxDepth=10 |
    | RandomForestRegressor | 700,212 | 49,768 | 0.0021 | numTrees=20 |
    | GBTRegressor | 702,852 | 48,466 | -0.0055 | maxIter=20 |

    Final test evaluation (touched once) for the selected `LinearRegression`: RMSE
    760,542, MAE 39,209, R² 0.0039. All output artifacts written correctly:
    `models/model_comparison.csv`, `models/model_metadata.json`,
    `models/model_metrics.json`, `models/best_salary_model/`.

    **Honest read of these numbers (this is the real finding, not just a log entry):**
    - **R² is ≈0 for every model** (max 0.0025, one model even negative) — essentially no
      better than predicting the mean salary for everyone. This isn't a bug; it matches
      the notebook prototype's R²≈0.012 too. These features just don't explain much
      variance in this self-reported salary data.
    - **The RMSE (~700K) vs. MAE (~43-49K) gap is extreme** (16x) — the math points
      squarely at outlier domination: this dataset's max salary is $74,351,432 (almost
      certainly a data-entry error, not real), and a single such value landing in a
      ~9,700-row validation/test split is enough on its own to produce an RMSE in this
      exact range. MAE (outlier-robust) is far more trustworthy here, and it's consistent
      with the notebook prototype's ~47-50K figures.
    - **Which model "won" is close to meaningless** — all four are statistically
      indistinguishable (R² within 0.008 of each other); `LinearRegression` won by noise,
      not a real edge.
    - **This is good evidence for revisiting the §10.9 outlier decision.** That section
      chose log1p-only (no hard filtering) based on the notebook prototype's results;
      this real run suggests log1p alone isn't containing the effect of extreme-value
      rows on RMSE. **Agreed next step (2026-08-11): implement real outlier handling**
      (capping and/or removing extreme salaries like the $74M row) as the next thing to
      work on, before doing anything else with Phase 4/5.
16. **Implemented outlier filtering (2026-08-11).** `src/training/data_cleaning.py`
    (§10.9) now drops rows with `label > MAX_PLAUSIBLE_SALARY` (default $1,000,000, new
    setting in `config/settings.py`/`.env.example`) before applying `log1p` to the rest —
    see the §10.9 note above for the full reasoning. Unit-tested
    (`test_clean_dataset_drops_implausibly_high_salaries`, confirming a row exactly at
    the cap is kept and rows above it are dropped), 7/7 `test_cleaning.py` tests passing.
17. **Outlier fix confirmed on the VM — it worked, decisively (2026-08-15).**
    `run_training_pipeline()` re-run against the real dataset with the fix in place.
    Only 60 rows dropped (48,016 → 47,956 cleaned rows, ~0.13%), for a dramatic result:

    | Model | RMSE (§23 #15 → now) | MAE (#15 → now) | R² (#15 → now) |
    |---|---|---|---|
    | LinearRegression (selected) | 700,079 → **59,158** | 43,702 → **31,319** | 0.0025 → **0.4623** |
    | DecisionTreeRegressor | 700,162 → 62,846 | 49,037 → 36,329 | 0.0022 → 0.3932 |
    | RandomForestRegressor | 700,212 → 65,401 | 49,768 → 37,526 | 0.0021 → 0.3428 |
    | GBTRegressor | 702,852 → 62,767 | 48,466 → 35,085 | -0.0055 → 0.3947 |

    Final test evaluation (touched once) for the selected `LinearRegression`: **RMSE
    58,323, MAE 30,429, R² 0.4585** — the model now explains ~46% of salary variance,
    a legitimate, respectable result for self-reported survey data with this feature set
    (R² in the 0.3–0.6 range is a realistic ceiling for this kind of noisy data, not a
    sign of a remaining problem). Confirms the RMSE-domination theory in #15 precisely:
    removing ~0.13% of rows (the most extreme outliers) cut RMSE by ~11-12x. Model
    selection is also now meaningful rather than noise — `LinearRegression` beats the
    next-best model (`GBTRegressor`, 62,767) by a real ~6% margin, not the near-identical
    numbers all 4 models had before. **This closes out the outlier-handling work someone
    flagged as the priority next step — Phase 4/5 is now in good shape both
    procedurally (leakage-free, real tuning, correct artifacts) and substantively (a
    real, working, moderately-accurate model).**
18. **Widened tuning grids + real concurrency for the VM's 16 cores (2026-08-15).**
    Explicitly requested: restore the grids/folds cut in #14 for speed, but keep it fast
    by actually using Spark/the VM's cores rather than reducing search breadth again.
    Three changes together:
    - **`SPARK_SHUFFLE_PARTITIONS`** (`config/settings.py`, default 16): applied to
      `spark.sql.shuffle.partitions`/`spark.default.parallelism` in `get_spark_session()`.
      Spark's own default (200) is tuned for large multi-node clusters; on this project's
      ~29K-row `cv_train` slice it means hundreds of tiny partitions and scheduling
      overhead instead of real parallelism. `tune_models.cross_validate()` also now
      explicitly `.repartition(SPARK_SHUFFLE_PARTITIONS)`s `data` and each constructed
      fold split, so every individual fit actually spreads across all 16 cores rather
      than whatever partition count fell out of prior `randomSplit`/`unionByName` calls.
    - **`TUNING_PARALLELISM`** (default 4): `cross_validate()` now runs this many
      (param combination, fold) fits *concurrently* via `concurrent.futures.
      ThreadPoolExecutor`, each wrapped with `pyspark.util.inheritable_thread_target` —
      the same mechanism `pyspark.ml.tuning.CrossValidator` uses internally, and the
      officially-documented way to submit Spark jobs from non-main threads. **This
      deliberately re-introduces the class of risk fixed in #13**
      (`ConnectionRefusedError` from background-thread py4j reconnection) — done
      knowingly, because the risk was judged worth it to actually use the VM's cores, and
      because `TUNING_PARALLELISM=1` reproduces the exact proven-reliable sequential
      behavior as an easy, documented fallback if it recurs.
    - **Grids widened** (`model_candidates.py`): each model now tunes 2 hyperparameters
      over ~6 combinations (was 1 hyperparameter/2 values), `NUM_FOLDS` back to 3 (was
      2) — 76 total pipeline fits (was 20). `RandomForestRegressor`/`GBTRegressor`
      ceilings deliberately stay below their original pre-OOM values
      (`numTrees`≤30/`maxDepth`≤8, `maxIter`≤30/`maxDepth`≤5) since concurrent fits mean
      multiple tree ensembles can be mid-training simultaneously, increasing peak memory
      pressure rather than reducing it — `SPARK_DRIVER_MEMORY` default also raised 4g→6g
      for the same reason.
    Also added `notebooks/run_training_pipeline.ipynb` — a ready-to-run VM notebook that
    stops any stale Spark session and explicitly prints actual-vs-expected config before
    letting the user proceed to training.
19. **Found and fixed running the above on the real VM (2026-08-15).** First attempt
    failed: `AnalysisException: Path does not exist: file:/home/linuxu/project/notebooks/
    data/raw/survey_results_public.csv`. Root cause: `os.chdir()` in the notebook only
    changes *Python's* working directory — the Spark JVM (a separate process reached via
    py4j) keeps whatever directory it was originally launched from (apparently
    `notebooks/`, wherever that persistent session first started), and resolves relative
    paths against *its own* cwd, not Python's. `DATASET_PATH`'s default
    (`./data/raw/survey_results_public.csv`) silently resolved against the wrong
    directory. Fixed at the root: every path setting in `config/settings.py`
    (`DATASET_PATH`, `MODEL_PATH`, `MODEL_METADATA_PATH`, `MODEL_COMPARISON_PATH`,
    `MODEL_METRICS_PATH`, the checkpoint paths) now resolves to an absolute path anchored
    to `PROJECT_ROOT`, computed from `settings.py`'s own file location — immune to either
    process's cwd. Same bug existed in `explore_dataset.py`'s `REPORT_DIR` (used in a
    Spark write, not just Python file I/O) and was fixed the same way. 11/11 relevant
    tests still passing after the change.
20. **Confirmed working on the VM, with improved results (2026-08-15).** Full run: 76
    fits across all 4 models completed in ~315s total — essentially the *same*
    wall-clock time the narrower 20-fit sequential run took before (~336s), confirming
    the concurrency actually delivered close to 4x the search for roughly the same time.
    No `ConnectionRefusedError`, no `OutOfMemoryError`.

    | | Before (§23 #17, narrow grid) | Now (wide grid + parallelism) |
    |---|---|---|
    | Validation RMSE | 59,158 | **56,770** |
    | Validation MAE | 31,319 | **30,376** |
    | Validation R² | 0.4623 | **0.4750** |
    | Final test RMSE | 58,323 | 59,425 |
    | Final test MAE | 30,429 | 31,024 |
    | Final test R² | 0.4585 | **0.4684** |

    `LinearRegression` still selected, but the wider search found a meaningfully
    different, better combination (`regParam=0.001, elasticNetParam=0.5` vs.
    `regParam=0.1` before) — validation metrics improved across the board; final test
    RMSE/MAE ticked up very slightly while R² improved, ordinary noise from evaluating on
    one fixed held-out sample rather than a red flag. Mean baseline: RMSE 78,352, R²≈0, as
    expected. **This closes out the explicit request to widen the grids using real
    Spark/VM capabilities rather than trading off search breadth for speed.**
21. **Kafka broker set up on the VM for the first time (2026-08-15).** `KAFKA_HOME` was
    still the `.env.example` placeholder (`/path/to/kafka`); real install found at
    `/usr/local/kafka/kafka_2.13-3.2.1` (matching Lab3's own `cdk` alias). ZooKeeper +
    broker started per Lab3's pattern (§6); all 5 required topics created
    (`salary_requests`, `salary_predictions`, `developer_events`, `salary_dead_letter`,
    `salary_analytics`). Confirmed reachable with the console producer/consumer. This is
    §24 Phase 1.8-1.10, previously unconfirmed.
22. **Found and fixed: the Kafka connector JAR can't be added to an already-running
    Spark session (2026-08-15).** `run_prediction_stream.ipynb`'s `build_streams()`
    call failed with `AnalysisException: Failed to find data source: kafka`, even after
    explicitly configuring `spark.jars.packages` via `get_spark_session(with_kafka=
    True)`. Diagnosis, confirmed step by step:
    - No Kafka connector JAR bundled anywhere under `$SPARK_HOME` (`find ... -iname
      "*kafka*.jar"` came back empty).
    - `spark.jars.packages` config *was* being correctly set to
      `org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0` — but `spark.jars` (actually
      resolved/loaded jars) stayed empty, and `~/.ivy2` had no cached Kafka jars (never
      successfully resolved).
    - Ruled out lack of internet access: `curl` to Maven Central returned HTTP 200.
    - Root cause, confirmed against a course-provided reference notebook ("Structured
      Streaming With Examples", shared for local troubleshooting only — not project
      content, not committed here), specifically its "Streaming from Kafka" section:
      `--packages` must be supplied when the PySpark **process
      itself** is launched (`pyspark --packages org.apache.spark:spark-sql-kafka-0-10_
      2.12:<version>`) — it cannot be injected into an already-running JVM via
      `.config(...)` from Python, no matter how the SparkSession is constructed. This
      explains every earlier symptom in this project (stop+recreate "fixing" the
      unrelated `ConnectionRefusedError` from §23 #13 but not this; `spark.driver.
      memory` not applying to a reused session, §23 #16/#20's whole "verify config"
      cell) — they're all the same underlying fact: JVM-level configuration is fixed at
      process launch and Python-level `.config()` calls on a reused session are largely
      cosmetic.
    - **Fix:** relaunch Jupyter itself through the `pyspark` launcher script so every
      notebook opened in that server inherits a Kafka-enabled session automatically:
      `PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser
      --port=8889' pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0`.
      No code changes were needed once this was done — `get_spark_session(with_kafka=
      False)`'s plain `getOrCreate()` correctly reuses the properly-launched session.
    - This is VM/environment-specific setup, not something `src/` code can fix or paper
      over — anyone else running this project needs to launch Jupyter the same way
      before any Kafka-touching notebook (`run_prediction_stream.ipynb`) will work.
23. **Phase 7 prediction stream confirmed working end-to-end against real Kafka
    (2026-08-15).** Via `notebooks/run_prediction_stream.ipynb`: published a valid
    request (Israel, back-end developer, 5 years experience, Bachelor's) to
    `salary_requests`, and `salary_predictions` correctly produced
    `{"request_id": "...", "prediction": 82046.35, "target_unit": "annual salary",
    "model_name": "LinearRegression", "model_version": "2026-08-15T17:11:53...",
    "status": "success"}` — a plausible number, correct shape, model name/version
    correctly pulled from the real `model_metadata.json`. Separately published a
    request with no `request_id`, confirmed it correctly landed on
    `salary_dead_letter` with `error_reason: "missing or unparseable request_id"`
    (twice, reproducibly). **This closes §24 Phase 7.5 for real** (previously only
    verified in the notebook prototype).

## 24. Step-by-Step Execution Checklist

We build this incrementally, one step at a time — each step is done, verified, and
confirmed before moving to the next. Nothing here executes yet; this is the checklist we
work through together. Checked items are done; unchecked items are next.

### Phase 1 — Repository, configuration, Spark and Kafka setup (VM, no Docker)
- [x] 1.1 Create `.gitignore`, `.env.example`, `requirements.txt`
- [x] 1.2 Folder skeleton from §7 created (`data/{raw,samples,processed}/`, `models/`,
      `checkpoints/`, `config/`, `src/{common,exploration,training,producers,streaming,
      dashboard}/`, `scripts/`, `tests/`), with `.gitkeep` placeholders matching
      `.gitignore`. This is now the single canonical project root — the notebook
      prototype from §23 still lives in the separate `big_data_project/` folder and has
      **not** been merged/ported in yet
- [x] 1.3 Extracted `survey_results_public.csv` from `data.zip` into `data/raw/`
      (2026-08-11, local to this dev machine only — correctly excluded from git by
      `.gitignore`; the VM and any other clone needs to run this same extraction step,
      it's not something a fresh checkout carries with it)
- [x] 1.4 `config/settings.py` — loads all §19 env vars via `python-dotenv` with the same
      defaults as `.env.example`; verified it imports and resolves defaults correctly
- [x] 1.5 `src/common/logging_config.py` — `get_logger(name)` with consistent formatting
- [x] 1.6 `src/common/spark_session.py` — `get_spark_session()` builds from
      `SPARK_MASTER_URL`/`SPARK_APP_NAME`, with an optional Kafka connector package for
      streaming jobs. Syntax-checked only — **pyspark isn't installed in this dev
      environment**, so it hasn't been run against a live Spark session yet
- [x] 1.7 `src/common/schemas.py` — explicit `StructType` schemas for `salary_requests`,
      `salary_predictions`, `developer_events`, `salary_dead_letter`, matching the §8
      message contracts (not yet reconciled with the notebook prototype's request shape,
      which added `LanguageCount`/`DatabaseCount`/`PlatformCount` instead of the raw
      skill strings — see Known Issues #3 in §23)
- [ ] 1.8 `scripts/start_kafka.sh` — not started (requires the actual VM)
- [ ] 1.9 `scripts/create_topics.sh` — not started (requires the actual VM)
- [ ] 1.10 Verify Kafka/Spark reachable on the VM — not done (no VM access from this
      environment)
- **Completion check:** Kafka broker and Spark are reachable, all topics exist — **not
  met**; 1.8–1.10 need an actual VM session

### Phase 2 — Dataset exploration and target decision
- [x] 2.1 `src/exploration/explore_dataset.py` covers §9 — schema, missing values,
      target-candidate-column check (stops with a `RuntimeError` instead of proceeding
      silently if `ConvertedCompYearly` is missing), cardinality, top
      languages/databases/platforms, exact-quantile salary diagnostics
- [x] 2.2 **Run against the real 89,184-row dataset on the actual VM (2026-08-11)** —
      via `build_report()` called directly in a Jupyter notebook, not yet via `main()`
      from a terminal, so the JSON/CSV file writes (`data/processed/`) haven't happened
      on the VM yet, only the in-memory report has been confirmed. This run is what
      surfaced and confirmed the fix for the `"NA"`-sentinel bug (§23 Known Issue #12):
      `rows_with_target_present: 48019, rows_with_target_missing: 41165`, matching §2
      exactly. Running `python -m src.exploration.explore_dataset` on the VM to produce
      the actual report files is the one remaining piece of this item
- [x] 2.3 Target-selection decision written back into this PLAN.md — see §2 "Final Target
      Decision" above (`ConvertedCompYearly`, 48,019/89,184 usable rows) — now confirmed
      twice: once via the notebook prototype, once via the ported script on the real VM
- **Completion check:** target report and data-quality report saved — **substantively
  met** (real numbers confirmed against the real dataset on the real VM); the on-disk
  report files just haven't been written there yet (trivial — `main()` vs. `build_report()`)

### Phase 3 — Cleaning and reusable feature pipeline
- [x] 3.1 `src/training/data_loader.py` — `load_raw_dataset(spark, path=None)`, reads
      `DATASET_PATH` from config with the same CSV options (`multiLine`, `escape='"'`)
      the prototype used
- [x] 3.2 `src/training/data_cleaning.py` — implements §10 steps 1–10 with row counts
      logged at every stage (§10.10, via `get_logger`). Two corrections over the
      notebook prototype: outlier quantiles are now computed exactly (not the broken
      `relError=0.01` approximation) and used for logging/diagnostics only, and the
      §10.9 outlier-vs-log1p decision is made explicit and documented in the module
      docstring (log1p chosen, `log_label` column added; no rows dropped for being high
      earners). `YearsCodeProNumeric` is deliberately left un-imputed here — see 3.3
- [x] 3.3 `src/training/feature_pipeline.py` — implements §11 in full: single-value
      `StringIndexer(handleInvalid='keep')` → `OneHotEncoder`; numeric `YearsCodeProNumeric`
      imputed via a Spark ML `Imputer` stage (median, fit only on the training fold —
      resolves the leakage shortcut the notebook's global pre-split imputation had);
      multi-value skill columns via `RegexTokenizer` + `CountVectorizer(binary=True)`
      building top-N (`TOP_LANGUAGES`/`TOP_DATABASES`/`TOP_PLATFORMS`, from
      `config/settings.py`) binary indicators fit only on training data — this replaces
      the notebook's `LanguageCount`/`DatabaseCount`/`PlatformCount` simplification
      (Known Issue #3 in §23, now resolved). Vocabulary reuse for streaming is automatic:
      it's embedded in the saved `PipelineModel`, no separate metadata file needed
- [x] 3.4 `tests/test_cleaning.py` (5 tests), `tests/test_features.py` (3 tests),
      `tests/test_schema.py` (4 tests) — **12/12 passing**, run locally against real
      PySpark (installed for this purpose; see §23 Known Issue #6). Caught and fixed a
      real bug in the process: ANSI-mode cast behavior (§23 Known Issue #7)
- **Completion check:** transformations pass unit tests — **met**, 12/12 passing locally
  (not yet re-run against the VM's exact Spark 3.3.0)

### Phase 4 — Train and tune candidate models
- [x] 4.1 `src/training/model_candidates.py` — all 4 required models (LinearRegression,
      DecisionTreeRegressor, RandomForestRegressor, GBTRegressor), each tuning one
      hyperparameter over 2 values (others fixed via constructor) — reduced from ~4
      combinations/2 hyperparameters after real runtime/memory problems on the VM (§23
      Known Issues #13/#14); "sized for the available machine" per §12 in practice, not
      just in theory. Widen again once a run comfortably completes with room to spare
- [x] 4.2 `src/training/data_split.py` + `tune_models.py` — real k-fold cross-validation
      (`NUM_FOLDS=2`) per model, fit only on the `cv_train` slice (60% of the cleaned
      dataset), tuned against log-space RMSE. Implemented as a hand-rolled sequential loop
      rather than `pyspark.ml.tuning.CrossValidator` — see Known Issue #13. This is the
      fix for Known Issues #1 and #2 in §23
- [x] 4.3 `tune_models.tune_all_candidates()` produces the comparison data (real-scale
      RMSE/MAE/R² per model, evaluated on the `validation` slice — data the CV step never
      saw) plus `evaluate_mean_baseline()` for the §13 naive baseline;
      `evaluate_model.write_model_comparison()` writes it to the exact
      `models/model_comparison.csv` path/schema §12 specifies
- **Completion check:** comparison table produced — **met, and confirmed against the real
  89,184-row dataset on the VM**: all 4 models tuned successfully. First run (2026-08-11,
  §23 #15) showed R² near zero for every model, traced to outlier domination; after the
  §10.9 outlier fix (§23 #17) and the subsequent widened-grid/parallel-tuning run (§23
  #20), best validation R² is now 0.4750 — a legitimate, working, tuned result

### Phase 5 — Select winner and final test evaluation
- [x] 5.1 `src/training/select_best_model.py` — RMSE → MAE → R² tie-break rule from §12,
      with a relative-tolerance tie check. Unit-tested directly, including both tie-break
      branches (`tests/test_select_best_model.py`); confirmed on real data —
      `LinearRegression` selected both before and after the outlier fix, now by a real
      ~6% margin over the next-best model rather than statistically-indistinguishable noise
- [x] 5.2 `src/training/train_final_model.py` — refits the selected model type +
      hyperparameters on `cv_train + validation` combined, evaluates **exactly once** on
      the untouched `test` slice, reverses `log1p` via `expm1` with negative/NaN clipping
      (§13). This closes the "evaluate once on untouched test data" gap — Known Issue #1
      in §23 is now resolved. Confirmed on real data, latest run after the outlier fix
      and widened tuning grids (§23 #20): final test RMSE 59,425 / MAE 31,024 / R² 0.4684
- [x] 5.3 `src/training/evaluate_model.py` — writes `models/model_metadata.json` and
      `models/model_metrics.json` in the exact §12 schema and saves the full
      `PipelineModel` via `.write().overwrite().save(...)` to `models/best_salary_model/`.
      **Confirmed on the real VM run (2026-08-11):** all 4 output artifacts written
      correctly against the real 48,016-row cleaned dataset, not just the earlier
      dummy-data smoke test
- [ ] 5.4 `scripts/train_and_select_model.sh` wiring Phases 3–5 together — not started
      (the Python entry point exists and works: `python -m src.training.evaluate_model`
      / `run_training_pipeline()`; the shell wrapper around it for the VM is what's
      missing)
- **Completion check:** best PipelineModel and metadata saved — **met and confirmed
  against real data, with a legitimately working, tuned model (R²=0.4684 final test)**
  after the outlier fix (§23 #17) and widened-grid parallel tuning (§23 #20). Only 5.4's
  shell wrapper remains outstanding — everything else in Phase 4/5 is done, tested, and
  validated against real data.

### Phase 6 — Kafka dataset producer
- [ ] 6.1 `src/producers/dataset_producer.py` — reads CSV gradually, adds `event_id`/
      `event_time`, publishes to `developer_events` with configurable delay, retries +
      delivery logging, graceful shutdown
- [ ] 6.2 `scripts/start_dataset_producer.sh` (file/topic/delay CLI args)
- [ ] 6.3 Verify with a console consumer (Lab3-style `kafka-console-consumer.sh
      --topic developer_events --from-beginning`)
- **Completion check:** events appear in `developer_events`

### Phase 7 — Prediction and analytics streams
- [x] 7.1 `src/streaming/prediction_stream.py` (2026-08-15) — ports and fixes the
      notebook prototype's gaps: reads `salary_requests` via `SALARY_REQUEST_SCHEMA`
      (now consistent with `feature_pipeline.py`'s actual raw-column expectations,
      unlike the notebook's stale `LanguageCount`-style schema), applies the saved
      `PipelineModel`, reverses `log1p` via the shared `reverse_log1p_predictions()`
      helper, publishes to `salary_predictions` keyed by `request_id`. **Adds what the
      notebook was missing:** `salary_dead_letter` routing for requests missing
      `request_id` (§16 step 8), `'Unknown'`-filling for missing non-required fields
      (avoids crashing `RegexTokenizer` on nulls), `model_name`/`model_version` read
      from the real `model_metadata.json` instead of hardcoded, real
      `checkpoints/salary_predictions/` + `checkpoints/salary_dead_letter/` (new
      `DEAD_LETTER_CHECKPOINT_PATH` setting) instead of a Spark temp dir, and a
      SIGTERM/SIGINT handler for graceful shutdown (§16 step 9). Pure transformation
      logic (`parse_requests`, `split_valid_and_dead_letters`, `build_predictions`,
      `to_kafka_rows`) is unit-tested against batch DataFrames. **Confirmed against a
      real Kafka broker on the VM (2026-08-15, §23 #23)**: a valid request correctly
      produced a real prediction (`$82,046.35`, correct response shape, real
      `model_name`/`model_version`); Kafka itself had to be set up on the VM for the
      first time to test this (§23 #21), and a genuinely important environment issue
      had to be found and fixed first (§23 #22 — the Kafka connector JAR can only be
      added at PySpark process launch, not via `.config()` on a running session;
      `notebooks/run_prediction_stream.ipynb` documents the fix)
- [ ] 7.2 `src/streaming/analytics_stream.py` — not started (no `developer_events`
      producer or aggregation logic exists yet)
- [ ] 7.3 `scripts/start_prediction_stream.sh` — not started (Python entry point exists
      and confirmed working: `python -m src.streaming.prediction_stream`, or
      interactively via `notebooks/run_prediction_stream.ipynb`)
- [ ] 7.4 `src/producers/prediction_request_producer.py` — not started as a standalone
      script, but `notebooks/run_prediction_stream.ipynb` now has a working
      `confluent_kafka`-based test-request publisher (§5 of that notebook) that could be
      extracted into one
- [x] 7.5 **Confirmed on the real VM against a real Kafka broker (2026-08-15).**
      Published a valid request, confirmed a matching prediction appeared on
      `salary_predictions` with the correct shape and a plausible value; published an
      invalid request (no `request_id`), confirmed it correctly landed on
      `salary_dead_letter` instead (reproduced twice) — see §23 #23 for the full
      request/response JSON
- **Completion check:** Spark consumes and publishes valid results — **met and
  confirmed against real Kafka**, both the happy path and the dead-letter path. Only
  the analytics stream (§15, Phase 7.2) remains unstarted within this phase

### Phase 8 — Dashboard
- [ ] 8.1 `src/dashboard/app.py` — personal prediction page (§17): inputs for all
      features, UUID per request, publish/await by `request_id`, timeout handling,
      estimate disclaimer
- [ ] 8.2 Descriptive analytics page: salary by country/experience/role/technology,
      salary distribution, Kafka event count
- [ ] 8.3 Run the dashboard end-to-end against the running stack
- **Completion check:** end-to-end request returns a prediction

### Phase 9 — Testing and documentation
- [ ] 9.1 Remaining tests: `test_model_selection.py`, `test_prediction_flow.py`
      (including the end-to-end Kafka request/response integration test on a small
      sample)
- [ ] 9.2 `README.md` — setup, run instructions, architecture summary
- [ ] 9.3 Walk through §22 Acceptance Criteria and confirm each item
- **Completion check:** README and automated tests complete

**Next action:** Phase 1, step 1.1 — pending your go-ahead to start.
