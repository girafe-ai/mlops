#!/usr/bin/env python3
"""Load metadata into PostgreSQL and materialize a SQL-filtered manifest."""

from pathlib import Path

import hydra
from cityscapes_tools.postgres import (
    build_connection_string,
    export_filtered_manifest,
    load_metadata,
    write_summary,
)
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(config: DictConfig) -> None:
    """Rebuild local PostgreSQL tables and export the configured image selection."""
    connection_string = build_connection_string(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password_env=config.database.password_env,
    )
    metadata_counts = load_metadata(
        connection_string=connection_string,
        schema_path=PROJECT_ROOT / "sql" / "schema.sql",
        metadata_dir=Path(config.data.metadata_dir),
    )
    labels = list(config.preprocess.labels)
    selected_images = export_filtered_manifest(
        connection_string=connection_string,
        query_path=PROJECT_ROOT / "sql" / "select_filtered_images.sql",
        labels=labels,
        output_path=Path(config.data.filtered_manifest_path),
    )
    write_summary(
        output_path=Path(config.data.summary_path),
        metadata_counts=metadata_counts,
        labels=labels,
        selected_images=selected_images,
    )
    print(f"Materialized {selected_images} selected images in PostgreSQL and CSV.")


if __name__ == "__main__":
    main()
