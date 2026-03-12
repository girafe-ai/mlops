"""
Generate fake events using Faker and store them in PostgreSQL.
Run as a script (one-off or scheduled) to populate the events table.
"""

import os
import sys
import time

import psycopg2
from faker import Faker
from psycopg2.extras import execute_batch

# Config from env
DB_HOST = os.environ.get("EVENTS_DB_HOST", "postgres")
DB_PORT = int(os.environ.get("EVENTS_DB_PORT", "5432"))
DB_NAME = os.environ.get("EVENTS_DB_NAME", "events")
DB_USER = os.environ.get("EVENTS_DB_USER", "airflow")
DB_PASSWORD = os.environ.get("EVENTS_DB_PASSWORD", "airflow")
BATCH_SIZE = int(os.environ.get("GENERATOR_BATCH_SIZE", "1000"))
NUM_BATCHES = int(os.environ.get("GENERATOR_NUM_BATCHES", "1"))

EVENT_TYPES = ("click", "view", "purchase", "login", "signup")
DEVICES = ("web", "mobile", "tablet")


def get_conn(dbname=None):
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname or DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def wait_for_postgres():
    """Wait until Postgres is reachable."""
    for _ in range(30):
        try:
            conn = get_conn()
            conn.close()
            return
        except Exception as e:
            print(f"Waiting for Postgres... ({e})")
            time.sleep(2)
    print("Postgres not available after 30 attempts.")
    sys.exit(1)


def main():
    """Insert fake events into the events table.

    Run create-events-db.sh first.
    """
    wait_for_postgres()
    fake = Faker()
    conn = get_conn()

    total = 0
    for batch in range(NUM_BATCHES):
        rows = []
        for _ in range(BATCH_SIZE):
            rows.append(
                (
                    fake.date_time_between(start_date="-7d", end_date="now"),
                    fake.random_element(EVENT_TYPES),
                    fake.random_element(DEVICES),
                    fake.random_int(min=1, max=100),
                    round(fake.random.uniform(0, 100), 2),
                )
            )
        with conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO events (ts, type, device, user_id, value)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
        total += len(rows)
        print(f"Inserted batch {batch + 1}/{NUM_BATCHES} ({len(rows)} rows)")
    conn.close()
    print(f"Done. Total rows inserted: {total}")


if __name__ == "__main__":
    main()
