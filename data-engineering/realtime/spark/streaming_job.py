"""
PySpark Structured Streaming job: read from Kafka topic 'events',
aggregate by event type in 10-second windows, print to console.
Runs for STREAM_TIMEOUT_SEC then exits (so Airflow task can complete).
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructType,
)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "events")
# Run for this many seconds then stop (for DAG demo)
STREAM_TIMEOUT_SEC = int(os.environ.get("STREAM_TIMEOUT_SEC", "60"))


def main():
    spark = (
        SparkSession.builder.appName("KafkaStreamingSeminar")
        .config(
            "spark.sql.streaming.checkpointLocation",
            "/tmp/spark-kafka-checkpoint",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    schema = (
        StructType()
        .add("ts", StringType())
        .add("type", StringType())
        .add("device", StringType())
        .add("user_id", IntegerType())
        .add("value", DoubleType())
    )
    events = df.select(
        F.from_json(F.col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Window 10 seconds, aggregate by event type
    windowed = (
        events.withColumn("ts", F.to_timestamp("ts"))
        .withWatermark("ts", "30 seconds")
        .groupBy(F.window("ts", "10 seconds"), "type")
        .agg(
            F.count("*").alias("count"),
            F.sum("value").alias("total_value"),
        )
        .orderBy("window", "type")
    )

    def foreach_batch(batch_df, batch_id):
        batch_df.show(truncate=False)

    query = (
        windowed.writeStream.outputMode("complete").foreachBatch(foreach_batch).start()
    )

    # Run for STREAM_TIMEOUT_SEC then stop (so Airflow task completes)
    query.awaitTermination(STREAM_TIMEOUT_SEC)
    query.stop()
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
