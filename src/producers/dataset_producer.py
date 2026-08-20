
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
    return value is None or value.strip() in _MISSING_TEXT_VALUES


def _parse_years_code_pro(value: str | None) -> float | None:
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
