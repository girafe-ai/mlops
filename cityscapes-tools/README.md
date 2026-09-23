# cityscapes-tools

## About Cityscapes

[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) contains urban
street scenes from 50 cities for semantic and instance segmentation. It includes
5,000 finely annotated images and 20,000 coarsely annotated images. See the
[annotation examples](https://www.cityscapes-dataset.com/examples/) for a preview.

## Downloading Cityscapes

[Register for a Cityscapes account](https://www.cityscapes-dataset.com/register/)
and accept the [dataset terms](https://www.cityscapes-dataset.com/license/)
before downloading. Create your local credentials file before running the downloader.

### Local credentials

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Fill in your Cityscapes credentials in the local `.env` file:

```dotenv
CITYSCAPES_USERNAME=your-email-or-username
CITYSCAPES_PASSWORD=your-password
```

Check the downloader's command and defaults without contacting Cityscapes or changing
files:

```bash
uv run python scripts/download_cityscapes.py download --dry-run
```

Download the standard left images and fine annotations to `data/cityscapes`:

```bash
uv run python scripts/download_cityscapes.py download
```

Pass other comma-separated package names from your Cityscapes download page when
needed:

```bash
uv run python scripts/download_cityscapes.py download \
  --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
  --destination=data/cityscapes
```

Use `--resume` to continue an interrupted download. The downloader verifies each
completed file against the MD5 checksum supplied by Cityscapes.

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
