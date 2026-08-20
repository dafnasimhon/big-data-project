
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
from src.common.feature_config import MULTI_VALUE_COLUMNS, SINGLE_VALUE_CATEGORICAL_COLUMNS

MULTI_VALUE_VOCAB_SIZES = {
    "Employment": settings.TOP_EMPLOYMENT_STATUSES,
    "LanguageHaveWorkedWith": settings.TOP_LANGUAGES,
    "DatabaseHaveWorkedWith": settings.TOP_DATABASES,
    "PlatformHaveWorkedWith": settings.TOP_PLATFORMS,
}


def build_feature_stages() -> list:
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

    multi_value_vector_columns = []
    for column in MULTI_VALUE_COLUMNS:
        token_column = f"{column}_tokens"
        vector_column = f"{column}_vec"
        stages.append(
            RegexTokenizer(inputCol=column, outputCol=token_column, pattern=";", toLowercase=False)
        )
        stages.append(
            CountVectorizer(
                inputCol=token_column,
                outputCol=vector_column,
                vocabSize=MULTI_VALUE_VOCAB_SIZES[column],
                binary=True,
            )
        )
        multi_value_vector_columns.append(vector_column)

    stages.append(
        VectorAssembler(
            inputCols=encoded_columns + ["YearsCodeProNumeric_imputed"] + multi_value_vector_columns,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    return stages


def build_feature_pipeline() -> Pipeline:
    return Pipeline(stages=build_feature_stages())
