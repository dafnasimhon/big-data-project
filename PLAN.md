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

> **Current actual layout (2026-08-11).** The skeleton exists (`data/{raw,samples,
> processed}/`, `models/`, `checkpoints/`, `config/settings.py`, `scripts/`, `tests/` —
> see Phase 1 in §24), and Phases 2–5 have real, tested code in it:
> `src/common/{logging_config,spark_session,schemas,feature_config,spark_utils}.py`,
> `src/exploration/explore_dataset.py`, `src/training/{data_loader,data_cleaning,
> feature_pipeline,data_split,model_candidates,tune_models,select_best_model,
> train_final_model,evaluate_model}.py`, plus 24 passing pytest tests across
> `tests/test_{cleaning,features,schema,model_candidates,select_best_model,
> training_pipeline}.py`. `src/{producers,streaming,dashboard}/` are still empty
> packages — Phase 7 (streaming) logic still only exists as the notebook prototype in the
> separate `big_data_project/` folder (`notebooks/`, `data/`) described in §23, not yet
> ported in. The raw CSV is now extracted into this repo's `data/raw/` (§24 Phase 1.3,
> local-only/gitignored), but an actual end-to-end run against it is blocked on this
> Windows dev machine by a missing `winutils.exe` (§23 Known Issue #10) — so none of the
> ported exploration/cleaning/training code has been run against the real 89,184-row file
> yet, only against small in-memory/synthetic samples in tests. That real-data run is the
> next validation step, on the VM.

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
11. **Full run against the real dataset is blocked on this Windows dev machine, not by
    code.** `data/raw/survey_results_public.csv` was extracted from `data.zip` here
    (2026-08-11) and `src/exploration/explore_dataset.py` was attempted against the real
    89,184-row file — it failed with `HADOOP_HOME and hadoop.home.dir are unset`.
    Any Spark operation that *writes files* (the exploration CSV report, and
    `PipelineModel.save()` in `evaluate_model.py`) needs Hadoop's `winutils.exe` on
    Windows; it isn't installed here. This doesn't affect the target Linux VM at all (no
    winutils needed there) and isn't a code defect — but it means **no part of Phases
    2–5 has actually been run against the real dataset yet**, only against small
    synthetic/in-memory samples in tests. This is the next real validation step, to be
    done on the VM (or after installing `winutils.exe` locally, if that's ever wanted for
    faster local iteration).

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
- [x] 2.1 Ported to `src/exploration/explore_dataset.py` (2026-08-11): a real,
      importable Spark job (`python -m src.exploration.explore_dataset`) covering §9 —
      schema, missing values, target-candidate-column check (stops with a `RuntimeError`
      instead of proceeding silently if `ConvertedCompYearly` is missing), cardinality,
      top languages/databases/platforms, and exact-quantile salary diagnostics. Its pure
      functions are smoke-tested against a tiny in-memory DataFrame; the full job has
      **not yet been run against the real 89,184-row CSV** since that file hasn't been
      copied into `data/raw/` in this repo yet (blocked on 1.3)
- [ ] 2.2 Run against the real dataset and confirm the report — pending 1.3 (CSV not yet
      in `data/raw/` here). Writes to `data/processed/exploration_report.json` +
      `data/processed/missing_values_summary/` once run (correct path per §7, unlike the
      notebook prototype's `output/`)
- [x] 2.3 Target-selection decision written back into this PLAN.md — see §2 "Final Target
      Decision" above (`ConvertedCompYearly`, 48,019/89,184 usable rows, from the
      notebook prototype's run against the real file)
- **Completion check:** target report and data-quality report saved — met by the
  notebook prototype's run; the ported script exists and is unit-tested but hasn't
  produced its own report yet (needs 1.3 first)

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
- [x] 4.1 `src/training/model_candidates.py` (2026-08-11) — all 4 required models
      (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor),
      each with a real `ParamGridBuilder` grid (~4 combinations, deliberately small —
      "sized for the available machine" per §12; widen on the VM)
- [x] 4.2 `src/training/data_split.py` + `tune_models.py` (2026-08-11) — real
      `CrossValidator(numFolds=3)` per model, fit only on the `cv_train` slice (60% of the
      cleaned dataset; see `data_split.py`), tuned against log-space RMSE. This is the
      fix for Known Issues #1 and #2 in §23
- [x] 4.3 `tune_models.tune_all_candidates()` produces the comparison data (real-scale
      RMSE/MAE/R² per model, evaluated on the `validation` slice — data CrossValidator
      never saw) plus `evaluate_mean_baseline()` for the §13 naive baseline;
      `evaluate_model.write_model_comparison()` writes it to the exact
      `models/model_comparison.csv` path/schema §12 specifies
- **Completion check:** comparison table produced — **met**, with real tuning. Code is
  unit-tested (`tests/test_model_candidates.py`, `tests/test_training_pipeline.py`) but
  **not yet run against the real 89,184-row dataset** (blocked on 1.3)

### Phase 5 — Select winner and final test evaluation
- [x] 5.1 `src/training/select_best_model.py` (2026-08-11) — RMSE → MAE → R² tie-break
      rule from §12, with a relative-tolerance tie check (floats essentially never match
      exactly). Unit-tested directly, including both tie-break branches
      (`tests/test_select_best_model.py`)
- [x] 5.2 `src/training/train_final_model.py` (2026-08-11) — refits the selected model
      type + hyperparameters on `cv_train + validation` combined, evaluates **exactly
      once** on the untouched `test` slice, reverses `log1p` via `expm1` with negative/NaN
      clipping (§13). This closes the "evaluate once on untouched test data" gap — Known
      Issue #1 in §23 is now resolved
- [x] 5.3 `src/training/evaluate_model.py` (2026-08-11) — writes
      `models/model_metadata.json` and `models/model_metrics.json` in the exact §12
      schema (selected model, selection metric, best params, validation + test metrics,
      target column/transformation, feature version, timestamp) and saves the full
      `PipelineModel` via `.write().overwrite().save(...)` to `models/best_salary_model/`.
      Verified with a dummy-data smoke test producing well-formed CSV/JSON; not yet run
      end-to-end against real data
- [ ] 5.4 `scripts/train_and_select_model.sh` wiring Phases 3–5 together — not started
      (the Python entry point exists: `python -m src.training.evaluate_model`; the shell
      wrapper around it for the VM is what's missing)
- **Completion check:** best PipelineModel and metadata saved — **met** by the code path
  (`evaluate_model.run_training_pipeline()`); not yet exercised against the real dataset,
  and 5.4's shell wrapper is still outstanding

### Phase 6 — Kafka dataset producer
- [ ] 6.1 `src/producers/dataset_producer.py` — reads CSV gradually, adds `event_id`/
      `event_time`, publishes to `developer_events` with configurable delay, retries +
      delivery logging, graceful shutdown
- [ ] 6.2 `scripts/start_dataset_producer.sh` (file/topic/delay CLI args)
- [ ] 6.3 Verify with a console consumer (Lab3-style `kafka-console-consumer.sh
      --topic developer_events --from-beginning`)
- **Completion check:** events appear in `developer_events`

### Phase 7 — Prediction and analytics streams
- [x] 7.1 Core logic proven end-to-end in `notebooks/04_Spark_Streaming_Prediction.ipynb`:
      reads `salary_requests` via Structured Streaming with an explicit schema, applies
      the saved `PipelineModel`, reverses `log1p` via `expm1`, publishes to
      `salary_predictions`. **Gaps before this counts as done:** no `salary_dead_letter`
      handling, no required-field validation, uses a Spark-generated temp checkpoint
      rather than `checkpoints/salary_predictions/`. Not yet extracted to
      `src/streaming/prediction_stream.py`
- [ ] 7.2 `src/streaming/analytics_stream.py` — not started (no `developer_events`
      producer or aggregation logic exists yet)
- [ ] 7.3 `scripts/start_prediction_stream.sh` — not started
- [ ] 7.4 `src/producers/prediction_request_producer.py` — not started; test requests were
      published manually/ad hoc during the notebook run
- [x] 7.5 Manually published requests and confirmed matching predictions appeared (visible
      in the notebook's `salary_prediction_results` in-memory table, keyed by
      `request_id`)
- **Completion check:** Spark consumes and publishes valid results — met for the
  prediction stream in prototype form; analytics stream (Phase 15/§15) not started

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
