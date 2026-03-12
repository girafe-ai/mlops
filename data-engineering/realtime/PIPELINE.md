# Pipeline Architecture & Component Guide

This document explains each part of the real-time streaming pipeline: what it
is, how it works, and how the pieces fit together.

---

## Overview

The pipeline is a **real-time analytics demo** that:

1. **Generates** sample events with Faker and stores them in **PostgreSQL**
2. **Pushes** events from Postgres to Kafka (reads only, no generation)
3. **Streams** them through PySpark Structured Streaming
4. **Aggregates** events by type in 10-second windows
5. **Orchestrates** the run on a schedule via Apache Airflow

![Pipeline scheme](assets/pipeline.png)

---

## 1. Zookeeper

**What it is:** A distributed coordination service used by Kafka to manage
cluster metadata, leader election, and configuration.

**How it works:**

- Kafka relies on Zookeeper to track broker membership, topic configurations,
  and consumer group offsets.
- Runs as a separate container, exposed on port **2181**.
- All services use the Docker network `seminar_net`, so Kafka connects via
  hostname `zookeeper:2181`.

**Why it's here:** Kafka (especially older versions / Confluent platform)
requires Zookeeper as a dependency. Newer Kafka versions support KRaft mode (no
Zookeeper), but this setup uses the classic architecture.

---

## 2. Kafka

**What it is:** A distributed **message broker** (event streaming platform) that
stores and delivers messages in topics.

**How it works:**

- **Topic:** `events` — a logical channel where the pusher writes and the Spark
  job reads. Create manually after `docker compose down`.
- **Broker:** One Kafka broker listens on `9092` (advertised as `kafka:9092` for
  containers).
- Messages are JSON objects: `{ ts, type, device, user_id, value }`.

**Key configuration:**

- `KAFKA_LISTENERS`: Where Kafka accepts connections (0.0.0.0:9092)
- `KAFKA_ADVERTISED_LISTENERS`: Host/port clients use to connect (`kafka:9092`
  on the Docker network)
- `KAFKA_ZOOKEEPER_CONNECT`: Zookeeper for cluster coordination

**Topic creation:** After each `docker compose down`, create the topic manually:
`docker compose exec kafka kafka-topics --create --topic events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1`

**Why it's here:** Kafka is the central event bus. The pusher pushes events;
Spark consumes them in streaming mode. Kafka buffers data and allows multiple
consumers to read at different speeds.

---

## 3. Generator

**What it is:** A Python script that uses the **Faker** library to generate fake
events and stores them in PostgreSQL.

**How it works:**

- Uses Faker for realistic fake data (timestamps, IDs, etc.).
- Connects to the `events` database in Postgres (run `create-events-db.sh`
  first).
- Inserts batches of rows (configurable via `GENERATOR_BATCH_SIZE` and
  `GENERATOR_NUM_BATCHES`).
- Event schema: `ts`, `type`, `device`, `user_id`, `value`, `sent_to_kafka`
  (flag for pusher).
- Runs once and exits. Invoked via `seed-data.sh`.

**Files:**

- `generator/generate_data.py` — main logic
- `generator/Dockerfile` — Python 3.11 + Faker + psycopg2

**Why it's here:** Produces sample data for the pipeline. Data is stored in
Postgres first, then the pusher reads and streams to Kafka.

---

## 4. Pusher

**What it is:** A Python service that reads events from PostgreSQL and pushes
them to the Kafka topic `events`. **No data generation**—only reading and
publishing.

**How it works:**

- Polls Postgres for rows where `sent_to_kafka = FALSE`.
- Fetches a batch (configurable via `PUSHER_BATCH_SIZE`).
- Sends each event as JSON to Kafka.
- Marks rows as `sent_to_kafka = TRUE`.
- Uses `FOR UPDATE SKIP LOCKED` for safe concurrent access.
- Runs continuously (poll interval configurable via `PUSHER_POLL_INTERVAL_SEC`).

**Files:**

- `pusher/pusher.py` — main logic
- `pusher/Dockerfile` — Python 3.11 + kafka-python + psycopg2

**Why it's here:** Decouples data generation (Postgres) from streaming (Kafka).
The pusher reads whatever is in the database and streams it—no generation logic.

---

## 5. Spark Master & Worker

**What it is:** Apache Spark cluster in standalone mode — one master and one
worker for distributed computation.

**Spark Master:**

- Coordinates the cluster and receives job submissions.
- Exposes Web UI on port **8080** and the driver submission port on **7077**.
- Has the Spark job code mounted at `/opt/spark/job` and the Ivy cache for Kafka
  connector JARs.

**Spark Worker:**

- Executes tasks assigned by the master.
- Connects to the master via `spark://spark-master:7077`.

**How it works:**

- When the Airflow DAG runs (or you run `submit-spark-job.sh`), a separate Spark
  container submits a job to the master.
- The master schedules the job on the worker; the driver runs inside the
  submission container.
- The `spark-sql-kafka` connector JARs are downloaded on first run and cached in
  the `spark_ivy_cache` volume.

**Why it's here:** Spark powers the streaming aggregation. Structured Streaming
reads from Kafka, applies windowing and aggregation, and outputs results.

---

## 6. PySpark Streaming Job (`streaming_job.py`)

**What it is:** The core processing logic — a PySpark Structured Streaming
application that reads from Kafka, aggregates by event type in 10-second
windows, and prints to the console.

**How it works:**

1. **Source:** Reads from Kafka topic `events` via
   `spark.readStream.format("kafka")`.

   - `startingOffsets: earliest` — reads all messages from the beginning of the
     topic (so existing data is processed).

2. **Parsing:** Raw Kafka values are JSON. The job uses `from_json` with a
   schema to extract `ts`, `type`, `device`, `user_id`, `value`.

3. **Windowing:** Groups events into **10-second tumbling windows** with a
   **30-second watermark** (for late-arriving data):

   ```python
   .withWatermark("ts", "30 seconds")
   .groupBy(F.window("ts", "10 seconds"), "type")
   ```

4. **Aggregation:** For each (window, event_type), computes:

   - `count` — number of events
   - `total_value` — sum of `value`

5. **Output:** Uses `foreachBatch` to print each micro-batch to the console.

6. **Lifecycle:** Runs for `STREAM_TIMEOUT_SEC` (default 60s in the DAG, 120s in
   the script), then stops. This lets the Airflow task complete instead of
   running forever.

**Checkpointing:** Uses `/tmp/spark-kafka-checkpoint` so Spark can resume from
the last committed offset on restart.

---

## 7. Airflow

**What it is:** An **orchestration platform** for scheduling and running
workflows (DAGs).

**How it works:**

- Runs in **standalone** mode: webserver, scheduler, and LocalExecutor in one
  container.
- Uses **PostgreSQL** for metadata (DAG runs, task instances, etc.).
- Connects to the host Docker daemon via `/var/run/docker.sock` to spawn Spark
  containers.
- The DAG `kafka_spark_streaming` runs on `@daily` schedule (or on manual
  trigger).

**DAG: `kafka_spark_streaming`**

- **Task:** `submit_spark_streaming` — runs a container with the Spark image.
  Data is pre-loaded via `create-events-db.sh` and `seed-data.sh` before
  starting the pipeline.
- **Command:** `spark-submit` with the streaming job.
- **Mounts:**
  - Host `./spark` → `/job` (read-only) — the streaming job code
  - Volume `tmp_spark_ivy` → `/home/spark/.ivy2` — Ivy cache for dependencies
- **Network:** `seminar_net` — so the Spark container can reach Kafka and the
  Spark master.

**Configuration notes:**

- `hostname: airflow` and `HOSTNAME=airflow` — fix log serving ("Could not read
  served logs") so the webserver can resolve the worker hostname.
- `AIRFLOW_SEMINAR_HOST_PATH` — project root path for mounting the Spark code
  (must be set to `$PWD` when running docker compose).

**Why it's here:** Orchestrates when and how the streaming job runs. In
production, you'd typically schedule it (e.g., daily) or trigger it from other
pipelines.

---

## 8. PostgreSQL

**What it is:** A relational database with two roles:

- **`airflow` database** — Airflow metadata (DAG runs, task instances,
  connections, variables).
- **`events` database** — Generated events from the Faker script (created by
  init script or generator).

**How it works:**

- Runs as a separate container with a persistent volume `postgres_data`.
- The `events` database is created by `create-events-db.sh` (run before
  `seed-data.sh`).
- Airflow connects via `postgresql+psycopg2://airflow:airflow@postgres/airflow`.
- Generator and pusher connect to `postgres/events` for the events table.
- Healthcheck ensures Postgres is ready before dependent services start.

**Why it's here:** Airflow requires a database backend; the `events` DB is the
staging area for generated data before it flows to Kafka.

---

## 9. Scripts

### `create-events-db.sh`

**What it is:** Creates the `events` database and table in Postgres.

**How it works:** Starts Postgres if needed, creates the database and table. Run
this **first** before `seed-data.sh`.

### `seed-data.sh`

**What it is:** Runs the Faker generator to insert fake events into the `events`
table.

**How it works:** Executes `docker compose run --rm generator`. Run **after**
`create-events-db.sh`.

### `submit-spark-job.sh`

**What it is:** A convenience script to run the Spark streaming job manually
(without Airflow).

**How it works:**

- Ensures Ivy cache directories exist (one-time setup).
- Runs a one-off container with the same image, mounts, and env vars as the
  Airflow task.
- Accepts optional timeout in seconds (default 120).
- Uses `docker run --rm` so the container is removed when done.

**When to use:** For quick testing or debugging without triggering the DAG.

---

## Data Flow Summary

| Step | Component                | Input                  | Output                                   |
| ---- | ------------------------ | ---------------------- | ---------------------------------------- |
| 1    | create-events-db.sh      | —                      | `events` database and table              |
| 2    | seed-data.sh (generator) | —                      | Fake events → Postgres `events` table    |
| 3    | Pusher                   | Postgres rows (unsent) | JSON events → Kafka `events` topic       |
| 4    | Kafka                    | Pusher events          | Buffered messages                        |
| 5    | Spark streaming job      | Kafka messages         | Parsed events → windowed aggregates      |
| 6    | Console                  | Aggregated batches     | Printed output (visible in Airflow logs) |
| 7    | Airflow                  | Schedule/trigger       | Runs Spark job only                      |

---

## Network & Connectivity

All services are on the Docker network `seminar_net`. They resolve each other by
service name:

- `kafka:9092` — Kafka broker
- `zookeeper:2181` — Zookeeper
- `spark-master:7077` — Spark master (job submission)
- `postgres` — PostgreSQL

The Airflow container uses `network_mode: bridge` (default) but spawns Spark
containers with `network_mode: seminar_net` so they can reach Kafka and the
Spark master.

---

## Volumes

| Volume            | Purpose                                                 |
| ----------------- | ------------------------------------------------------- |
| `postgres_data`   | Airflow metadata (DAG runs, tasks)                      |
| `spark_ivy_cache` | Cached Spark/Kafka connector JARs (avoids re-download)  |
| `tmp_spark_ivy`   | Same cache, used by the Airflow-spawned Spark container |

---

## Ports (Host)

| Port | Service                           |
| ---- | --------------------------------- |
| 2181 | Zookeeper                         |
| 5432 | Postgres                          |
| 7077 | Spark master (driver submission)  |
| 8080 | Spark master Web UI               |
| 8081 | Airflow Web UI (mapped from 8080) |
| 9092 | Kafka                             |
