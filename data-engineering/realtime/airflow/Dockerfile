# Airflow + Docker provider so we can submit Spark jobs via DockerOperator
FROM apache/airflow:2.7.3

USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir apache-airflow-providers-docker
