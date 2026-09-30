# Downloading Cityscapes locally

Register for a [Cityscapes account](https://www.cityscapes-dataset.com/register/)
and accept the [dataset terms](https://www.cityscapes-dataset.com/license/).
From the repository root, enter the tools directory and create a local credentials
file from the template:

```bash
cd cityscapes-tools
cp .env.example .env
```

Fill in `CITYSCAPES_USERNAME` and `CITYSCAPES_PASSWORD` in the local `.env` file.
Then check the downloader's defaults without contacting Cityscapes or creating
data files:

```bash
uv run python scripts/download_cityscapes.py download --dry-run
```

Download the standard left images and fine annotations:

```bash
uv run python scripts/download_cityscapes.py download
```

The command saves `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
under `cityscapes-tools/data/cityscapes/`. It verifies each archive's MD5 checksum
but does not extract the archives. Use `--resume` to continue an interrupted
download.

To choose other packages or a destination, use names from your Cityscapes download
page:

```bash
uv run python scripts/download_cityscapes.py download \
  --packages=leftImg8bit_trainvaltest.zip \
  --destination=data/cityscapes
```
