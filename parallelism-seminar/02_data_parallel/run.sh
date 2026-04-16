#!/usr/bin/env bash
# Run DDP and FSDP at world sizes 2 and 4 to populate metrics.csv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
LAUNCH="${ROOT}/00_setup/launch.sh"

for NGPUS in 2 4; do
  echo "=== DDP  world_size=${NGPUS} ==="
  "${LAUNCH}" "${NGPUS}" "${HERE}/train_ddp.py"

  echo "=== FSDP world_size=${NGPUS} ==="
  "${LAUNCH}" "${NGPUS}" "${HERE}/train_fsdp.py"
done
