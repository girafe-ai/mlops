# Seminar: Realtime PySpark Streaming + Kafka + Airflow

Everything runs in **Docker**—no local Kafka, PySpark, or Airflow install
required.

## Quick start

### 1. Create database

```bash
./scripts/create-events-db.sh
```

### 2. Insert data

```bash
./scripts/seed-data.sh
```

### 3. Start pipeline

```bash
docker compose up -d
```

### 4. Create Kafka topic (required after each `docker compose down`)

```bash
docker compose exec kafka kafka-topics --create --topic events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 5. Reset sent flag (required after each `docker compose down`)

Kafka loses its messages on restart, but Postgres remembers them as already sent.
Reset the flag so the pusher re-sends everything:

```bash
docker compose exec postgres psql -U airflow -d events -c "UPDATE events SET sent_to_kafka = FALSE;"
```

Give Airflow ~1–2 minutes to init. Then open http://localhost:8081 (admin /
admin), unpause the **kafka_spark_streaming** DAG, and trigger it.

## Architecture

- **Generator** – Faker script stores fake events in Postgres (run via
  seed-data.sh)
- **Pusher** – Reads Postgres, pushes to Kafka topic `events`
- **Kafka** – Message broker
- **PySpark** – Reads Kafka, aggregates by event type in 10s windows
- **Airflow** – DAG submits Spark streaming job

## Project layout

```
.
├── docker-compose.yml
├── generator/
├── pusher/
├── spark/
├── airflow/
├── scripts/
│   ├── create-events-db.sh
│   ├── seed-data.sh
│   └── submit-spark-job.sh
└── README.md
```

## Troubleshooting

- **Airflow login fails** – The `postgres_data` volume persists between restarts,
  so the password update on startup can silently fail. Reset the admin user:
  ```bash
  docker compose exec -u airflow airflow airflow users delete --username admin
  docker compose exec -u airflow airflow airflow users create \
    --username admin --password admin \
    --firstname Admin --lastname User \
    --role Admin --email admin@example.com
  ```
- **Empty table in Spark output / no data in stream** – After `docker compose down`,
  Kafka loses all messages but Postgres still marks rows as `sent_to_kafka = TRUE`.
  The pusher finds nothing to send → Spark reads an empty topic → empty table.
  Fix: recreate the topic (step 4) and reset the flag (step 5).
- **UnknownTopicOrPartitionException** – Create the topic:
  `docker compose exec kafka kafka-topics --create --topic events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1`
