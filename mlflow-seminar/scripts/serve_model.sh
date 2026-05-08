#!/usr/bin/env bash
set -euo pipefail

MODEL_URI="${MODEL_URI:-models:/cityscapes-segmentation@champion}"
PORT="${PORT:-5001}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"

export MLFLOW_TRACKING_URI="${TRACKING_URI}"

mlflow models serve \
  -m "${MODEL_URI}" \
  -p "${PORT}" \
  --env-manager local
