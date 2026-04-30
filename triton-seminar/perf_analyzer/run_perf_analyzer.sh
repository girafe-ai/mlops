#!/usr/bin/env bash
set -u

URL="${URL:-triton:8000}"
PROTOCOL="${PROTOCOL:-http}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-1:8:1}"
MEASUREMENT_INTERVAL_MS="${MEASUREMENT_INTERVAL_MS:-5000}"
RESULTS_DIR="${RESULTS_DIR:-perf_analyzer/results}"
DOCKER_COMPOSE="${DOCKER_COMPOSE:-sudo docker compose}"

mkdir -p "$RESULTS_DIR"

run_perf() {
  local model_name="$1"
  local batch_size="$2"
  local shape="$3"
  local output_csv="$RESULTS_DIR/${model_name}_perf.csv"

  echo "Running perf_analyzer for model=${model_name}, batch_size=${batch_size}, shape=${shape}"
  $DOCKER_COMPOSE run --rm perf-analyzer \
    -m "$model_name" \
    -u "$URL" \
    -i "$PROTOCOL" \
    --concurrency-range "$CONCURRENCY_RANGE" \
    --measurement-interval "$MEASUREMENT_INTERVAL_MS" \
    --input-data random \
    --shape "$shape" \
    -b "$batch_size" \
    -f "$output_csv"
}

run_perf cityscapes 1 input:1,3,1024,2048
run_perf cityscapes_dyn 1 input:3,1024,2048

echo "Perf Analyzer CSVs written to $RESULTS_DIR"
