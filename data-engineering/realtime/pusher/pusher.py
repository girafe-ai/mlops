"""
Read events from PostgreSQL (no generation) and push them to Kafka.
Runs continuously, polling for unsent rows and publishing to the Kafka topic.
"""

import json
import os
import time

import psycopg2
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from psycopg2.extras import RealDictCursor

# Config from env
DB_HOST = os.environ.get("EVENTS_DB_HOST", "postgres")
DB_PORT = int(os.environ.get("EVENTS_DB_PORT", "5432"))
DB_NAME = os.environ.get("EVENTS_DB_NAME", "events")
DB_USER = os.environ.get("EVENTS_DB_USER", "airflow")
DB_PASSWORD = os.environ.get("EVENTS_DB_PASSWORD", "airflow")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "events")
POLL_INTERVAL_SEC = float(os.environ.get("PUSHER_POLL_INTERVAL_SEC", "2.0"))
BATCH_SIZE = int(os.environ.get("PUSHER_BATCH_SIZE", "100"))


def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def wait_for_kafka():
    while True:
        try:
            p = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            p.close()
            return
        except NoBrokersAvailable:
            print("Waiting for Kafka...")
            time.sleep(3)


def event_row_to_json(row):
    """Convert DB row to JSON matching Spark job schema.

    Schema: ts, type, device, user_id, value.
    """
    ts = row["ts"]
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat() + "Z" if ts.tzinfo is None else ts.isoformat()
    return {
        "ts": ts,
        "type": row["type"],
        "device": row["device"],
        "user_id": int(row["user_id"]),
        "value": float(row["value"]),
    }


def main():
    wait_for_kafka()
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(
        f"Pusher started: Postgres -> Kafka topic '{TOPIC}', "
        f"poll every {POLL_INTERVAL_SEC}s"
    )

    while True:
        try:
            conn = get_db_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, ts, type, device, user_id, value
                    FROM events
                    WHERE sent_to_kafka = FALSE
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (BATCH_SIZE,),
                )
                rows = cur.fetchall()

            if not rows:
                conn.close()
                time.sleep(POLL_INTERVAL_SEC)
                continue

            ids = [r["id"] for r in rows]
            for row in rows:
                msg = event_row_to_json(row)
                producer.send(TOPIC, value=msg)
            producer.flush()

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE events SET sent_to_kafka = TRUE WHERE id = ANY(%s)",
                    (ids,),
                )
            conn.commit()
            conn.close()
            print(f"Pushed {len(rows)} events to Kafka")

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
