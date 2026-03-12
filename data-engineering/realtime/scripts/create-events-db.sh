#!/usr/bin/env bash
# Create the events database and table. Run this BEFORE seed-data.sh.
# Requires: Postgres (start with: docker compose up -d postgres)
#
# Usage: ./scripts/create-events-db.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Ensuring Postgres is running..."
docker compose up -d postgres

echo "Waiting for Postgres to be ready..."
for i in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U airflow 2>/dev/null; then
    break
  fi
  sleep 2
done

echo "Creating events database (if not exists)..."
docker compose exec -T postgres psql -U airflow -d postgres -c "SELECT 1 FROM pg_database WHERE datname='events'" | grep -q 1 \
  || docker compose exec -T postgres psql -U airflow -d postgres -c "CREATE DATABASE events"

echo "Creating events table (if not exists)..."
docker compose exec -T postgres psql -U airflow -d events -c "
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    type VARCHAR(50) NOT NULL,
    device VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    sent_to_kafka BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"

echo "Done. Database 'events' and table 'events' are ready."
