#!/usr/bin/env bash
# Insert fake events into the events table. Run create-events-db.sh first.
#
# Usage: ./scripts/seed-data.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

docker compose build generator 2>/dev/null || true
docker compose run --rm generator
