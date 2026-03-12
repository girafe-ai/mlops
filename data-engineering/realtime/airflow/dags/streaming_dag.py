"""
Seminar DAG: PySpark Streaming from Kafka.
- Data is pre-loaded via scripts (create-events-db.sh, seed-data.sh).
- Pusher runs as a service (Postgres → Kafka).
- submit_spark_streaming: PySpark reads from Kafka, aggregates by event type
  in 10s windows, runs for 60s then exits.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

_HOST_PATH = os.environ.get("AIRFLOW_SEMINAR_HOST_PATH", ".")
_NETWORK = os.environ.get("AIRFLOW_SEMINAR_NETWORK", "seminar_net")
_SPARK_PACKAGES = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

with DAG(
    dag_id="kafka_spark_streaming",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["seminar", "streaming", "kafka", "spark"],
) as dag:
    submit_spark_streaming = DockerOperator(
        task_id="submit_spark_streaming",
        image="apache/spark:3.5.0",
        api_version="auto",
        auto_remove="success",
        network_mode=_NETWORK,
        mount_tmp_dir=False,
        environment={
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "KAFKA_TOPIC": "events",
            "STREAM_TIMEOUT_SEC": "60",
        },
        command=[
            "/opt/spark/bin/spark-submit",
            "--master",
            "spark://spark-master:7077",
            "--packages",
            _SPARK_PACKAGES,
            "/job/streaming_job.py",
        ],
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(
                target="/job",
                source=f"{_HOST_PATH}/spark",
                type="bind",
                read_only=True,
            ),
            Mount(
                target="/home/spark/.ivy2",
                source="tmp_spark_ivy",
                type="volume",
            ),
        ],
    )
