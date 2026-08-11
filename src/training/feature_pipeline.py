"""Implements PLAN.md §11 feature engineering as a reusable, leakage-free Spark ML Pipeline.

All stages here are Estimators (StringIndexer, OneHotEncoder, Imputer, CountVectorizer)
fit only when `Pipeline.fit(train_df)` is called on the training fold — never on
validation/test data (PLAN.md rule 7). The fitted vocabularies/indices are embedded in
the resulting PipelineModel when it's saved, which is what §11 step 6 means by "save the
exact vocabulary in model metadata for identical streaming transformations": the saved
PipelineModel *is* that metadata, so streaming inference just loads and applies it rather
than re-deriving the vocabulary.

Deviation from the notebook prototype (documented in PLAN.md §23 "Known issues" #3): the
prototype used raw skill-list length (`LanguageCount` etc.) instead of per-skill
indicators. This version implements §11 as written — `RegexTokenizer` splits each
`;`-separated skill field, then `CountVectorizer(binary=True)` builds a top-N
(`TOP_LANGUAGES`/`TOP_DATABASES`/`TOP_PLATFORMS`) binary indicator vector per field, fit
from the training data only. Values outside the top-N are simply dropped (no explicit
"other" bucket) — that part of §11 is optional and this keeps the pipeline to Spark ML's
built-in transformers rather than a custom one.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    CountVectorizer,
    Imputer,
    OneHotEncoder,
    RegexTokenizer,
    StringIndexer,
    VectorAssembler,
)

from config import settings
from src.common.feature_config import SINGLE_VALUE_CATEGORICAL_COLUMNS, SKILL_COLUMNS

SKILL_COLUMN_VOCAB_SIZES = {
    "LanguageHaveWorkedWith": settings.TOP_LANGUAGES,
    "DatabaseHaveWorkedWith": settings.TOP_DATABASES,
    "PlatformHaveWorkedWith": settings.TOP_PLATFORMS,
}


def build_feature_stages() -> list:
    """Build the (unfit) §11 Spark ML stages, ending in a `features` vector column."""
    stages = []

    indexed_columns = []
    for column in SINGLE_VALUE_CATEGORICAL_COLUMNS:
        indexed_column = f"{column}_index"
        stages.append(
            StringIndexer(inputCol=column, outputCol=indexed_column, handleInvalid="keep")
        )
        indexed_columns.append(indexed_column)

    encoded_columns = [f"{column}_encoded" for column in SINGLE_VALUE_CATEGORICAL_COLUMNS]
    stages.append(
        OneHotEncoder(inputCols=indexed_columns, outputCols=encoded_columns, handleInvalid="keep")
    )

    stages.append(
        Imputer(
            inputCols=["YearsCodeProNumeric"],
            outputCols=["YearsCodeProNumeric_imputed"],
            strategy="median",
        )
    )

    skill_vector_columns = []
    for column in SKILL_COLUMNS:
        token_column = f"{column}_tokens"
        vector_column = f"{column}_vec"
        stages.append(
            RegexTokenizer(inputCol=column, outputCol=token_column, pattern=";", toLowercase=False)
        )
        stages.append(
            CountVectorizer(
                inputCol=token_column,
                outputCol=vector_column,
                vocabSize=SKILL_COLUMN_VOCAB_SIZES[column],
                binary=True,
            )
        )
        skill_vector_columns.append(vector_column)

    stages.append(
        VectorAssembler(
            inputCols=encoded_columns + ["YearsCodeProNumeric_imputed"] + skill_vector_columns,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    return stages


def build_feature_pipeline() -> Pipeline:
    """A fresh, unfit feature-engineering Pipeline — fit it once per candidate model
    training run inside its own Pipeline (with the regressor appended), never shared
    across models after fitting, per §12's leakage-prevention requirement."""
    return Pipeline(stages=build_feature_stages())
