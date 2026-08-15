#!/usr/bin/env bash
# Launches Jupyter through the `pyspark` command itself, with the Kafka connector
# package pre-loaded at process launch time - the only point at which it can be added
# (see PLAN.md §23 #22 for the full story: a running Spark session can't pick up a new
# connector JAR via .config() after the fact, so this has to happen before Jupyter, and
# therefore any notebook's SparkSession, ever starts).
#
# Any notebook opened through the Jupyter server this starts (e.g.
# notebooks/run_prediction_stream.ipynb) will have a working Kafka-enabled Spark session
# available automatically via SparkSession.builder.getOrCreate() - no special
# configuration needed in the notebook itself.
#
# Usage:
#   scripts/start_kafka_jupyter.sh [port]
#
# port defaults to 8889 (not 8888) so this doesn't collide with a Jupyter server you
# might already have running without Kafka support.

set -euo pipefail

PORT="${1:-8889}"

if ! command -v pyspark >/dev/null 2>&1; then
    echo "ERROR: 'pyspark' is not on PATH. Set SPARK_HOME/bin on PATH first (see .env), e.g.:" >&2
    echo "  export PATH=\"\$SPARK_HOME/bin:\$PATH\"" >&2
    exit 1
fi

PYSPARK_VERSION="$(python3 -c 'import pyspark; print(pyspark.__version__)')"
KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:${PYSPARK_VERSION}"

echo "Starting Jupyter via pyspark on port ${PORT}, with ${KAFKA_PACKAGE}..."
echo "Open the URL/token pyspark prints below, then open notebooks/run_prediction_stream.ipynb through it."
echo

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=${PORT}"

exec pyspark --packages "${KAFKA_PACKAGE}"
