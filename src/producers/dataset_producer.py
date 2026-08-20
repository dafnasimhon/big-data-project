"""Kafka producer for `developer_events` (PLAN.md §14).

Replays the survey CSV row-by-row onto `developer_events`, at a configurable delay, to
simulate a live feed of incoming developer profiles rather than a static batch file.
Downstream, `notebooks/run_prediction_stream.ipynb`'s Section 8 reads these events back
(they carry the real `ConvertedCompYearly` from the source row) to compare the trained
model's predictions against real salaries. (An earlier windowed-aggregation consumer,
`src/streaming/analytics_stream.py`/Phase 7.2, was built, confirmed working, and then
dropped by decision on 2026-08-20 - this producer is unaffected either way.)

Deliberately plain Python, not Spark: reads the CSV with the standard library's `csv`
module, one row at a time, so memory use stays flat regardless of file size, and uses a
lightweight `confluent_kafka.Producer` - the same pattern already proven in
`notebooks/run_prediction_stream.ipynb`'s test-request publisher and Lab3's own producers.

Run with:

    python -m src.producers.dataset_producer
    python -m src.producers.dataset_producer --file path/to.csv --topic developer_events --delay 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

from config import settings
from src.common.feature_config import CANDIDATE_INPUT_FEATURES, TARGET_COLUMN
from src.common.logging_config import get_logger

logger = get_logger(__name__)

_MISSING_TEXT_VALUES = {"", "NA"}


def _is_missing(value: str | None) -> bool:
    """Same "NA"-as-missing-sentinel rule as `src/common/spark_utils.py: is_missing_text`,
    reimplemented in plain Python since this module deliberately has no Spark dependency."""
    return value is None or value.strip() in _MISSING_TEXT_VALUES


def _parse_years_code_pro(value: str | None) -> float | None:
    """Mirrors `data_cleaning.convert_years_code_pro`'s mapping so events carry the same
    numeric meaning the training pipeline gives this field, without importing Spark here."""
    if _is_missing(value):
        return None
    if value == "Less than 1 year":
        return 0.5
    if value == "More than 50 years":
        return 51.0
    try:
        return float(value)
    except ValueError:
        return None


def _parse_numeric(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def build_event(row: dict) -> dict:
    """Selects the relevant fields from one raw CSV row and adds `event_id`/`event_time`
    (PLAN.md §14). Numeric fields (`YearsCodePro`, target) are coerced to real numbers (or
    null) here rather than left as raw CSV strings, so the resulting JSON matches
    `src/common/schemas.py: DEVELOPER_EVENT_SCHEMA`'s declared types exactly - this doesn't
    depend on Spark's `from_json` string-to-number coercion behavior downstream.
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "event_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for field in CANDIDATE_INPUT_FEATURES:
        if field == "YearsCodePro":
            event[field] = _parse_years_code_pro(row.get(field))
        else:
            value = row.get(field)
            event[field] = None if _is_missing(value) else value
    event[TARGET_COLUMN] = _parse_numeric(row.get(TARGET_COLUMN))
    return event


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("Delivery failed for event %s: %s", msg.key(), err)
    else:
        logger.info(
            "Delivered event %s to %s[%d]@%d", msg.key(), msg.topic(), msg.partition(), msg.offset()
        )


def stream_dataset(
    csv_path: str,
    topic: str,
    delay_seconds: float,
    bootstrap_servers: str | None = None,
    limit: int | None = None,
) -> int:
    """Reads `csv_path` gradually and publishes one event per row to `topic`, waiting
    `delay_seconds` between events. Returns the number of events published. Stops early,
    gracefully, on SIGINT/SIGTERM (finishes the in-flight row, flushes, then returns).

    `limit`, if given, stops after publishing that many events - the full survey CSV
    (~89K rows) at any reasonable demo delay would take hours to replay in full, so a
    bounded run is the practical way to demo this interactively (e.g. from a notebook).
    """
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS,
            "retries": 5,
            "acks": "all",
        }
    )

    stop_requested = False

    def _request_stop(signum, frame):  # noqa: ARG001 - signal handler signature
        nonlocal stop_requested
        logger.info("Received shutdown signal, finishing current row then stopping...")
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    published = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if stop_requested or (limit is not None and published >= limit):
                break
            event = build_event(row)
            producer.produce(
                topic, key=event["event_id"], value=json.dumps(event), callback=_delivery_callback
            )
            producer.poll(0)
            published += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    producer.flush()
    logger.info("Published %d events to %s (stopped_early=%s).", published, topic, stop_requested)
    return published


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the dataset CSV onto a Kafka topic.")
    parser.add_argument("--file", default=settings.DATASET_PATH, help="Path to the CSV to replay.")
    parser.add_argument("--topic", default=settings.KAFKA_DATASET_TOPIC, help="Kafka topic to publish to.")
    parser.add_argument(
        "--delay",
        type=float,
        default=settings.DATASET_EVENT_DELAY_SECONDS,
        help="Seconds to wait between events.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after publishing this many events (default: replay the whole file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stream_dataset(args.file, args.topic, args.delay, limit=args.limit)


if __name__ == "__main__":
    main()
