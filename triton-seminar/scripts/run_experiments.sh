#!/usr/bin/env bash
set -u

METRICS_FILE="${METRICS_FILE:-metrics.txt}"
URL="${URL:-localhost:8000}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"

run_profile() {
  local model_name="$1"
  local concurrency="$2"
  local batch_size="$3"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  printf "\n===== model=%s concurrency=%s batch_size=%s timestamp=%s =====\n" \
    "$model_name" "$concurrency" "$batch_size" "$timestamp" >> "$METRICS_FILE"

  if python scripts/load_test_triton.py \
    --url "$URL" \
    --model-name "$model_name" \
    --concurrency "$concurrency" \
    --batch-size "$batch_size" \
    --num-requests "$NUM_REQUESTS"; then
    curl localhost:8002/metrics >> "$METRICS_FILE" || true
  else
    printf "load_test_failed model=%s concurrency=%s batch_size=%s\n" \
      "$model_name" "$concurrency" "$batch_size" >> "$METRICS_FILE"
  fi
}

for model_name in cityscapes cityscapes_dyn; do
  for concurrency in 1 4 8 16; do
    run_profile "$model_name" "$concurrency" 1
  done
done

for concurrency in 1 4 8 16; do
  run_profile cityscapes_dyn "$concurrency" 4
done
