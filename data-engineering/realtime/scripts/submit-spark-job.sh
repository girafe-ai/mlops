#!/usr/bin/env bash
# Submit the PySpark streaming job (one-off container on seminar_net).
# Usage: ./scripts/submit-spark-job.sh [timeout_seconds]
# First run may download Kafka connector JARs (~30s).

set -e
TIMEOUT="${1:-120}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure Ivy cache dirs exist (one-time; apache/spark image needs writable .ivy2)
docker run --rm -u root -v tmp_spark_ivy:/home/spark/.ivy2 --entrypoint "" apache/spark:3.5.0 \
  bash -c "mkdir -p /home/spark/.ivy2/cache /home/spark/.ivy2/jars && chown -R spark:spark /home/spark/.ivy2" 2>/dev/null || true

docker run --rm --network seminar_net \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e KAFKA_TOPIC=events \
  -e STREAM_TIMEOUT_SEC="$TIMEOUT" \
  -v "$PROJECT_ROOT/spark:/job:ro" \
  -v tmp_spark_ivy:/home/spark/.ivy2 \
  apache/spark:3.5.0 \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /job/streaming_job.py
