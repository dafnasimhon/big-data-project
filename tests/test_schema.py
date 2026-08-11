from pyspark.sql.types import StructType

from src.common.schemas import (
    DEAD_LETTER_SCHEMA,
    DEVELOPER_EVENT_SCHEMA,
    SALARY_PREDICTION_SCHEMA,
    SALARY_REQUEST_SCHEMA,
)


def test_schemas_are_struct_types():
    for schema in (
        SALARY_REQUEST_SCHEMA,
        SALARY_PREDICTION_SCHEMA,
        DEVELOPER_EVENT_SCHEMA,
        DEAD_LETTER_SCHEMA,
    ):
        assert isinstance(schema, StructType)


def test_salary_request_schema_has_required_fields():
    field_names = SALARY_REQUEST_SCHEMA.fieldNames()
    for required in ("request_id", "Country", "YearsCodePro"):
        assert required in field_names


def test_salary_prediction_schema_has_required_fields():
    field_names = SALARY_PREDICTION_SCHEMA.fieldNames()
    for required in ("request_id", "prediction", "status"):
        assert required in field_names


def test_developer_event_schema_has_target_column():
    assert "ConvertedCompYearly" in DEVELOPER_EVENT_SCHEMA.fieldNames()
    assert "event_id" in DEVELOPER_EVENT_SCHEMA.fieldNames()
