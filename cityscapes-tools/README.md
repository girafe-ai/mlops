# cityscapes-tools

For dataset details and download instructions, see [DATASET.md](DATASET.md).

## Local credentials

From the repository root, enter the tools directory:

```bash
cd cityscapes-tools
```

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Fill in your Cityscapes credentials in the local `.env` file:

```dotenv
CITYSCAPES_USERNAME=your-email-or-username
CITYSCAPES_PASSWORD=your-password
```

## Building and publishing

Build source and wheel distributions with:

```bash
uv build
```

To publish a build to PyPI, put a project-scoped PyPI API token in `.env` as
`UV_PUBLISH_TOKEN`, then run:

```bash
uv publish
```

## Downloading dataset

Run the following commands from the `cityscapes-tools` directory. Check the
downloader's defaults without contacting Cityscapes or changing files:

```bash
uv run python scripts/download_cityscapes.py download --dry-run
```

Download the standard left images and fine annotations:

```bash
uv run python scripts/download_cityscapes.py download
```

This saves `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` under
`cityscapes-tools/data/cityscapes/` in your local repository. The downloader
verifies each archive's MD5 checksum; it does not extract the archives.

Pass other comma-separated package names from your Cityscapes download page when
needed:

```bash
uv run python scripts/download_cityscapes.py download \
  --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
  --destination=data/cityscapes
```

Use `--resume` to continue an interrupted download.

## Download dataset

Run the following commands from the `cityscapes-tools` directory. Check the
downloader's defaults without contacting Cityscapes or changing files:

```bash
uv run python scripts/download_cityscapes.py download --dry-run
```

Download the standard left images and fine annotations:

```bash
uv run python scripts/download_cityscapes.py download
```

This saves `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` under
`cityscapes-tools/data/cityscapes/` in your local repository. The downloader
verifies each archive's MD5 checksum; it does not extract the archives.

Pass other comma-separated package names from your Cityscapes download page when
needed:

```bash
uv run python scripts/download_cityscapes.py download \
  --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
  --destination=data/cityscapes
```

Use `--resume` to continue an interrupted download.