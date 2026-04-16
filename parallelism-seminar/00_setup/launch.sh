#!/usr/bin/env bash
# Usage: ./launch.sh <NGPUS> <script.py> [script args...]
#
# Thin wrapper around torchrun for single-node multi-GPU jobs used in the
# seminar. Standalone rendezvous = no master addr/port bookkeeping.
set -euo pipefail

NGPUS="${1:?usage: launch.sh NGPUS script.py [args...]}"
SCRIPT="${2:?usage: launch.sh NGPUS script.py [args...]}"
shift 2

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  "${SCRIPT}" "$@"
