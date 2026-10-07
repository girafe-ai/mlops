"""Load Cityscapes metadata into PostgreSQL and export SQL-selected manifests."""

import csv
import json
import os
from pathlib import Path

import psycopg


def build_connection_string(
    host: str,
    port: int,
    database: str,
    user: str,
    password_env: str,
) -> str:
    """Build a local PostgreSQL connection string from a password environment variable."""
    password = os.getenv(password_env, "cityscapes")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def load_metadata(
    connection_string: str,
    schema_path: Path,
    metadata_dir: Path,
) -> dict[str, int]:
    """Recreate local tables and bulk-load generated CSV metadata.

    The database is deliberately a reproducible cache: each execution applies
    the schema and reloads the data from DVC-tracked inputs.
    """
    images_path = metadata_dir / "images.csv"
    objects_path = metadata_dir / "objects.csv"
    schema = schema_path.read_text(encoding="utf-8")

    with (
        psycopg.connect(connection_string) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(schema)
        cursor.execute("TRUNCATE objects, images;")
        _copy_csv(cursor, images_path, "images")
        _copy_csv(cursor, objects_path, "objects")
        cursor.execute("SELECT COUNT(*) FROM images;")
        image_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM objects;")
        object_count = cursor.fetchone()[0]

    return {"images": image_count, "objects": object_count}


def export_filtered_manifest(
    connection_string: str,
    query_path: Path,
    labels: list[str],
    output_path: Path,
) -> int:
    """Run the supplied class filter and write selected image records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = query_path.read_text(encoding="utf-8")

    with (
        psycopg.connect(connection_string) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(query, (labels,))
        rows = cursor.fetchall()

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image_id", "split", "city", "image_path", "annotation_path"])
        writer.writerows(rows)
    return len(rows)


def write_summary(
    output_path: Path,
    metadata_counts: dict[str, int],
    labels: list[str],
    selected_images: int,
) -> None:
    """Write a compact DVC metric describing the materialized selection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": metadata_counts,
        "selected_labels": labels,
        "selected_images": selected_images,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _copy_csv(cursor: psycopg.Cursor, path: Path, table_name: str) -> None:
    """Copy a headered UTF-8 CSV file into a PostgreSQL table."""
    with (
        path.open("r", encoding="utf-8") as source_file,
        cursor.copy(
            f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
        ) as copy,
    ):
        for line in source_file:
            copy.write(line)
