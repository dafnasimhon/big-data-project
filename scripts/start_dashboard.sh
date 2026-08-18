#!/usr/bin/env bash
# Thin CLI wrapper around `streamlit run src/dashboard/app.py` (PLAN.md §24 Phase 8.3).
#
# Unlike the streaming modules, the dashboard has no Spark dependency (it talks to Kafka
# directly via confluent_kafka), so this should work as a plain terminal command - no
# Kafka-connector-JAR launch concerns like scripts/start_kafka_jupyter.sh.
#
# Usage:
#   scripts/start_dashboard.sh [port]
#
# port defaults to 8501 (Streamlit's own default).

set -euo pipefail

PORT="${1:-8501}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec streamlit run src/dashboard/app.py --server.port "${PORT}"
