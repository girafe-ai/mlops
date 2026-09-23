# Cityscapes dataset

[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) contains urban
street scenes from 50 cities for semantic and instance segmentation. It includes
5,000 finely annotated images and 20,000 coarsely annotated images. See the
[annotation examples](https://www.cityscapes-dataset.com/examples/) for a preview.

## Downloading Cityscapes

[Register for a Cityscapes account](https://www.cityscapes-dataset.com/register/)
and accept the [dataset terms](https://www.cityscapes-dataset.com/license/)
before downloading. Set up local credentials as described in
[README.md](README.md#local-credentials).

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
