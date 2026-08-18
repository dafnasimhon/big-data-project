"""Streamlit dashboard (PLAN.md §17): personal salary prediction + descriptive analytics.

Deliberately has no Spark dependency - like `src/producers/dataset_producer.py`, this
talks to Kafka directly via `confluent_kafka`, so `streamlit run src/dashboard/app.py`
should work as a plain terminal command with no connector-JAR launch concerns.

Run with:

    streamlit run src/dashboard/app.py

Design choices worth documenting:
  - Feature inputs are free-text (with real example placeholders), not dropdowns/
    selectboxes populated from a fixed category list. Two reasons: (1) this project has
    no reliable source of the exact category strings without querying the real dataset
    live (the raw CSV isn't available outside the VM, and `explore_dataset.py`'s report
    only records cardinality *counts*, not the value strings themselves, for single-value
    categorical columns) - a hardcoded guess would risk being subtly wrong, and one
    already was (`RemoteWork` turned out to include values like "Hybrid (some remote,
    some in-person)", not just "Hybrid"). (2) It's low-stakes either way:
    `feature_pipeline.py`'s `StringIndexer(handleInvalid="keep")` and `CountVectorizer`
    both tolerate unrecognized values gracefully (treated as "unknown"/dropped, not a
    crash), so a slightly-off free-text value degrades a prediction's precision rather
    than breaking it.
  - `await_prediction()` uses a fresh, unique consumer group per request, reading from
    "earliest" rather than "latest" - see its docstring for why this avoids a
    subscribe-vs-publish race at the cost of re-scanning the topic each time (fine at
    this project's demo scale).
  - The analytics page's "salary distribution" is a histogram of *per-segment average*
    salaries from `salary_breakdown` (streaming aggregates), not a true per-individual
    histogram - `salary_analytics` never carries raw per-event salaries, only aggregates.
    Labeled as such in the UI rather than overclaiming precision.
"""

from __future__ import annotations

import json
import time
import uuid

import pandas as pd
import streamlit as st
from confluent_kafka import Consumer, Producer

from config import settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)

FIELD_PLACEHOLDERS = {
    "Country": "e.g. Israel",
    "Age": "e.g. 25-34 years old",
    "EdLevel": "e.g. Bachelor's degree",
    "Employment": "e.g. Employed, full-time",
    "RemoteWork": "e.g. Hybrid",
    "DevType": "e.g. Developer, back-end",
    "OrgSize": "e.g. 100 to 499 employees",
    "Industry": "e.g. Information Services, IT, Software Development",
    "LanguageHaveWorkedWith": "e.g. Python;SQL",
    "DatabaseHaveWorkedWith": "e.g. PostgreSQL",
    "PlatformHaveWorkedWith": "e.g. AWS",
}


def _none_if_blank(value: str) -> str | None:
    """Sends JSON `null` for an empty field rather than `""` - `split_valid_and_dead_
    letters()`'s `fillna("Unknown", ...)` on the consuming side only catches true nulls,
    not empty strings (PLAN.md §16 step 4)."""
    stripped = value.strip()
    return stripped if stripped else None


def publish_request(request: dict) -> None:
    producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
    producer.produce(
        settings.KAFKA_REQUEST_TOPIC, key=request["request_id"], value=json.dumps(request)
    )
    producer.flush()


def await_prediction(request_id: str, timeout_seconds: float) -> dict | None:
    """Waits for `request_id`'s matching response on `salary_predictions`.

    Uses a fresh, unique consumer group starting from "earliest" for every call, rather
    than "subscribe, then poll" on a shared group - the latter would race against
    exactly when the prediction stream publishes its response relative to when this
    consumer's group finishes subscribing. Re-scanning the topic from the start every
    call is wasteful once it holds a lot of history, but at this project's demo scale
    that's a fair trade for guaranteed correctness.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": f"dashboard-await-{request_id}",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([settings.KAFKA_PREDICTION_TOPIC])
    deadline = time.time() + timeout_seconds
    try:
        while time.time() < deadline:
            msg = consumer.poll(timeout=1.0)
            if msg is None or msg.error():
                continue
            payload = json.loads(msg.value())
            if payload.get("request_id") == request_id:
                return payload
    finally:
        consumer.close()
    return None


def read_topic_snapshot(topic: str, poll_seconds: float = 5.0) -> list[dict]:
    """Reads whatever is currently on `topic` from the beginning, within `poll_seconds` -
    same batch-consumer pattern as the notebooks' verification cells, used here for the
    analytics page's "Refresh" button."""
    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": f"dashboard-analytics-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    records = []
    deadline = time.time() + poll_seconds
    try:
        while time.time() < deadline:
            msg = consumer.poll(timeout=1.0)
            if msg is None or msg.error():
                continue
            records.append(json.loads(msg.value()))
    finally:
        consumer.close()
    return records


def _weighted_avg(rows: list[dict], group_key: str, value_key: str = "avg_salary") -> dict[str, float]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    return {
        key: sum(r[value_key] * r["event_count"] for r in group) / sum(r["event_count"] for r in group)
        for key, group in grouped.items()
    }


def render_personal_prediction() -> None:
    st.header("Personal Salary Prediction")
    st.caption(
        "This produces a survey-based statistical estimate, not a guarantee of your "
        "actual market salary."
    )
    st.info(
        "Free-text fields accept any value - unrecognized categories are treated as "
        "'unknown' rather than rejected, but predictions are most reliable when your "
        "wording matches the placeholders below (real examples from the training data)."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            country = st.text_input("Country", placeholder=FIELD_PLACEHOLDERS["Country"])
            age = st.text_input("Age", placeholder=FIELD_PLACEHOLDERS["Age"])
            ed_level = st.text_input("Education level", placeholder=FIELD_PLACEHOLDERS["EdLevel"])
            employment = st.text_input("Employment", placeholder=FIELD_PLACEHOLDERS["Employment"])
            remote_work = st.text_input("Remote work", placeholder=FIELD_PLACEHOLDERS["RemoteWork"])
            years_code_pro = st.number_input(
                "Years of professional coding experience",
                min_value=0.0, max_value=60.0, value=5.0, step=0.5,
            )
        with right:
            dev_type = st.text_input("Developer role", placeholder=FIELD_PLACEHOLDERS["DevType"])
            org_size = st.text_input("Organization size", placeholder=FIELD_PLACEHOLDERS["OrgSize"])
            industry = st.text_input("Industry", placeholder=FIELD_PLACEHOLDERS["Industry"])
            languages = st.text_input(
                "Languages you've worked with (';'-separated)",
                placeholder=FIELD_PLACEHOLDERS["LanguageHaveWorkedWith"],
            )
            databases = st.text_input(
                "Databases you've worked with (';'-separated)",
                placeholder=FIELD_PLACEHOLDERS["DatabaseHaveWorkedWith"],
            )
            platforms = st.text_input(
                "Cloud platforms you've worked with (';'-separated)",
                placeholder=FIELD_PLACEHOLDERS["PlatformHaveWorkedWith"],
            )
        submitted = st.form_submit_button("Get my prediction")

    if not submitted:
        return

    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "Country": _none_if_blank(country),
        "Age": _none_if_blank(age),
        "EdLevel": _none_if_blank(ed_level),
        "Employment": _none_if_blank(employment),
        "RemoteWork": _none_if_blank(remote_work),
        "YearsCodePro": years_code_pro,
        "DevType": _none_if_blank(dev_type),
        "OrgSize": _none_if_blank(org_size),
        "Industry": _none_if_blank(industry),
        "LanguageHaveWorkedWith": _none_if_blank(languages),
        "DatabaseHaveWorkedWith": _none_if_blank(databases),
        "PlatformHaveWorkedWith": _none_if_blank(platforms),
    }

    submitted_at = time.time()
    with st.spinner(
        f"Publishing request and waiting up to {settings.DASHBOARD_PREDICTION_TIMEOUT_SECONDS}s "
        "for a prediction..."
    ):
        publish_request(request)
        logger.info("Published prediction request %s", request_id)
        prediction = await_prediction(request_id, settings.DASHBOARD_PREDICTION_TIMEOUT_SECONDS)
    elapsed = time.time() - submitted_at

    if prediction is None:
        st.error(
            f"No prediction received within {settings.DASHBOARD_PREDICTION_TIMEOUT_SECONDS}s. "
            "Is the prediction stream running? (`notebooks/run_prediction_stream.ipynb`, PLAN.md §16)"
        )
    elif prediction.get("status") != "success":
        st.error(f"Request could not be processed: {prediction}")
    else:
        st.success(f"Estimated annual salary: ${prediction['prediction']:,.2f}")
        st.caption(
            f"Model: {prediction['model_name']} (trained {prediction['model_version']}) - "
            f"processed at {prediction['processed_at']} - round trip {elapsed:.2f}s"
        )
        st.caption(
            "Estimate only, based on the Stack Overflow Developer Survey 2023 - actual "
            "salaries vary by factors this model doesn't capture."
        )


def render_descriptive_analytics() -> None:
    st.header("Descriptive Analytics")
    st.caption(
        "Live snapshot of `salary_analytics` (PLAN.md §15). Needs "
        "`notebooks/run_dataset_producer.ipynb` and `notebooks/run_analytics_stream.ipynb` "
        "to have been run (or still be running) for there to be anything to show."
    )

    if st.button("Refresh") or "analytics_snapshot" not in st.session_state:
        with st.spinner(f"Reading {settings.KAFKA_ANALYTICS_TOPIC}..."):
            st.session_state["analytics_snapshot"] = read_topic_snapshot(settings.KAFKA_ANALYTICS_TOPIC)
    records = st.session_state["analytics_snapshot"]

    if not records:
        st.warning(
            "No analytics records found yet. Run the dataset producer and analytics "
            "stream notebooks, then click Refresh."
        )
        return

    by_metric: dict[str, list[dict]] = {}
    for record in records:
        by_metric.setdefault(record["metric"], []).append(record)

    event_counts = by_metric.get("event_counts", [])
    st.metric("Kafka events processed (developer_events)", sum(row["event_count"] for row in event_counts))

    breakdown = by_metric.get("salary_breakdown", [])
    if breakdown:
        st.subheader("Average salary by country")
        st.bar_chart(_weighted_avg(breakdown, "country"))

        st.subheader("Average salary by developer role")
        st.bar_chart(_weighted_avg(breakdown, "role"))

        st.subheader("Average salary by years of experience")
        st.bar_chart(_weighted_avg(breakdown, "experience_range"))

        st.subheader("Salary distribution")
        st.caption(
            "Histogram of per-segment average salaries (window x country x role x "
            "experience) - an approximation built from streaming aggregates, not a true "
            "per-individual histogram (the raw per-event salary isn't published)."
        )
        salary_df = pd.DataFrame({"avg_salary": [row["avg_salary"] for row in breakdown]})
        histogram = salary_df.groupby(pd.cut(salary_df["avg_salary"], bins=10)).size()
        histogram.index = histogram.index.astype(str)
        st.bar_chart(histogram)

    tech_counts = by_metric.get("technology_counts", [])
    if tech_counts:
        left, right = st.columns(2)
        with left:
            st.subheader("Technology popularity")
            grouped: dict[str, list[dict]] = {}
            for row in tech_counts:
                grouped.setdefault(row["technology"], []).append(row)
            st.bar_chart({tech: sum(r["event_count"] for r in rows) for tech, rows in grouped.items()})
        with right:
            st.subheader("Average salary by technology")
            st.bar_chart(_weighted_avg(tech_counts, "technology"))


def main() -> None:
    st.set_page_config(page_title="Tech Salary Prediction", layout="wide")
    st.title("Real-Time Tech Salary Prediction")

    page = st.sidebar.radio("Page", ["Personal Prediction", "Descriptive Analytics"])
    if page == "Personal Prediction":
        render_personal_prediction()
    else:
        render_descriptive_analytics()


if __name__ == "__main__":
    main()
