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
> see Phase 1 in §24), and Phases 2–3 have real, tested code in it:
> `src/common/{logging_config,spark_session,schemas,feature_config,spark_utils}.py`,
> `src/exploration/explore_dataset.py`, `src/training/{data_loader,data_cleaning,
> feature_pipeline}.py`, plus `tests/{test_cleaning,test_features,test_schema}.py`
> (12/12 passing locally). `src/{producers,streaming,dashboard}/` are still empty
> packages — Phase 4/5 (tuned training + selection) and Phase 7 (streaming) logic still
> only exists as the notebook prototype in the separate `big_data_project/` folder
> (`notebooks/`, `data/`) described in §23, not yet ported in. The raw CSV itself also
> hasn't been copied into this repo's `data/raw/` yet (§24 Phase 1.3), so the ported
> exploration/cleaning code hasn't been run end-to-end against the real 89,184-row file
> here — only against small in-memory samples in tests.

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
1. **Test-set leakage across notebooks.** 03, 03B, and 03C all reuse the *same* seed-42
   80/20 split, so the "test" set was used to (a) pick Random Forest as the winner, (b)
   decide the log1p + skill-count version was an improvement, and (c) report final KPIs —
   three decisions on data that's no longer an unbiased test set. Needs a real 3-way
   split: train/validation for all comparison and improvement decisions, one test set
   touched exactly once at the very end, per §12's data-splitting strategy. **Still open**
   — this is Phase 4/5 work, not touched by the Phase 2/3 porting done on 2026-08-11.
2. **No real hyperparameter tuning.** Each model uses one hardcoded parameter set; §12
   requires `CrossValidator`/`TrainValidationSplit` over the listed param grids. **Still
   open** — Phase 4 work.
3. ~~Multi-value skill features simplified to counts~~ **Resolved (2026-08-11).**
   `src/training/feature_pipeline.py` now implements §11 as written: `RegexTokenizer` +
   `CountVectorizer(binary=True)` per skill column, building top-N (`TOP_LANGUAGES`/
   `TOP_DATABASES`/`TOP_PLATFORMS`) binary indicators fit only on the training fold —
   see §24 Phase 3.
4. **Model quality is weak** — R² of 0.012 is barely above a flat baseline. Still an open
   question once real training happens against the ported pipeline; worth revisiting
   `Country`/`DevType` one-hot cardinality and whether per-skill indicators (now
   available, see #3) help more than the raw counts did.
5. **Output artifacts don't match §12's required paths/schema.** Still open — applies once
   Phase 4/5 training modules are ported; the exploration report now correctly writes to
   `data/processed/` (see §24 Phase 2), but `models/model_comparison.csv` /
   `model_metadata.json` / `model_metrics.json` don't exist yet.
6. ~~Not yet ported to `src/`~~ **Partially resolved (2026-08-11).** Phase 2 exploration
   and Phase 3 cleaning/feature-pipeline logic now live in `src/exploration/` and
   `src/training/`, config-driven via `config/settings.py`, with 12 passing pytest tests
   (`tests/test_cleaning.py`, `tests/test_features.py`, `tests/test_schema.py`) run
   locally against real PySpark (4.2.0 — not yet re-verified against the VM's pinned
   Spark 3.3.0; the APIs used are stable across that range, but treat as unverified on
   the actual target version until run there). Phase 4/5 training/selection and Phase 7
   streaming logic are **not yet ported**.
7. **New: found and fixed during porting.** The prototype's bare `.cast("double")` on
   non-numeric strings (e.g. `"NA"`) relies on Spark's ANSI SQL mode being off (the VM's
   Spark 3.3.0 default) to silently return null — under ANSI mode (the default in newer
   Spark) the same cast *raises an exception* instead. `src/common/spark_utils.py:
   safe_cast_double()` fixes this with a regex-validated cast that behaves the same
   either way; `data_cleaning.py` and `explore_dataset.py` both use it now. This was
   only caught because the ported code was actually tested against real PySpark rather
   than assumed correct from the notebook's observed behavior.

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
- [ ] 1.3 Extract `survey_results_public.csv` from `data.zip` into `data/raw/` — not done
      yet (the prototype instead reads its own copy of the CSV from
      `big_data_project/data/`)
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
- [x] 4.1 All 4 required models (LinearRegression, DecisionTreeRegressor,
      RandomForestRegressor, GBTRegressor) implemented and trained in
      `notebooks/03_Model_Training.ipynb` — **not yet extracted** to
      `src/training/model_candidates.py`, and each uses one hardcoded parameter set
      rather than the §12 parameter grids
- [ ] 4.2 **Not done.** 80/20 split (seed 42) exists, but no `CrossValidator`/
      `TrainValidationSplit` tuning was run — this is a real gap, not just a missing file;
      see "Known issues" #2 in §23
- [x] 4.3 Comparison table produced (`output/model_comparison`, CSV) — needs to move to
      `models/model_comparison.csv` per §12 once ported
- **Completion check:** comparison table produced — met for a single-parameter-set
  comparison; true tuning (4.2) still outstanding

### Phase 5 — Select winner and final test evaluation
- [x] 5.1 Selection-by-lowest-RMSE logic implemented inline in
      `notebooks/03_Model_Training.ipynb` (picked Random Forest) — MAE/R² tie-break
      branches not exercised (no ties occurred); not yet extracted to
      `src/training/select_best_model.py`
- [ ] 5.2 Refit + log1p reversal done in `03B_Model_Improvement.ipynb`, **but the "evaluate
      once on the untouched test set" requirement was violated** — the same test split
      was reused across 03/03B/03C for multiple decisions (Known issues #1 in §23). Needs
      a real held-out test set touched exactly once, plus porting to
      `src/training/train_final_model.py`
- [x] 5.3 `models/best_salary_model/` saved and verified reloadable; metrics/comparison
      exist as CSVs under `output/` — **not yet** `models/model_metadata.json` /
      `models/model_metrics.json` in the exact §12 schema (target column, transformation,
      feature version, etc.). Still needs `src/training/evaluate_model.py`
- [ ] 5.4 `scripts/train_and_select_model.sh` wiring Phases 3–5 together — not started
- **Completion check:** best PipelineModel and metadata saved — model itself is saved and
  reloadable; metadata format and the leakage-free test evaluation are still outstanding

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
