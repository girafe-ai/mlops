# DVC, Hydra, and PostgreSQL pipeline

This project treats Cityscapes files as the source of truth. DVC tracks the
raw annotation directory and the generated filtered manifest; PostgreSQL is a
local, rebuildable query layer rather than another data store to version.

## Start PostgreSQL

```bash
docker compose up -d
docker compose ps
```

The default local connection is `cityscapes:cityscapes@localhost:5432/cityscapes`.
Set `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, or `POSTGRES_PORT` in
your local `.env` before the first startup to replace those values.

## Run the pipeline

```bash
uv run dvc repro
```

The `metadata` stage turns `gtFine` polygon JSON into `data/metadata/*.csv`.
The `filter` stage recreates PostgreSQL tables, loads those CSV files, selects
images containing the configured labels, and writes:

- `data/processed/filtered_manifest.csv` — DVC-tracked selected image records;
- `reports/filtered_summary.json` — a DVC metric.

The active selection lives in `conf/preprocess/preprocess.yaml`. Override it
without editing a file:

```bash
uv run dvc repro --force
uv run python scripts/materialize_filtered_dataset.py \
  preprocess.labels=[truck,bus,train]
```

For a fresh database, recreate the container volume with:

```bash
docker compose down -v
docker compose up -d
```

Do not use this command when you need to retain other local PostgreSQL data.
