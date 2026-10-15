# cityscapes-tools

Cityscapes data preparation and segmentation training with raw PyTorch or Lightning.
For dataset details and official resources, see the [dataset guide](docs/dataset.md).
See the [training guide](docs/training.md) for the code structure and seminar walkthrough.
The guide also explains how to [run MLflow locally](docs/training.md#local-mlflow-logging)
and select Lightning's built-in logger with `logger=mlflow`.

## Install and commands

From `cityscapes-tools/`, install the Python 3.10 project environment:

```bash
uv sync --locked
```

Commands are registered in `pyproject.toml` and implemented in
`src/cityscapes_tools/cli/`. Run them from the directory containing your data:

| Command                             | Purpose                                        |
| ----------------------------------- | ---------------------------------------------- |
| `uv run cityscapes-download`        | Download Cityscapes archives                   |
| `uv run cityscapes-build-metadata`  | Extract annotation metadata                    |
| `uv run cityscapes-materialize`     | Load PostgreSQL and export a filtered manifest |
| `uv run cityscapes-train-torch`     | Start the explicit PyTorch loop                |
| `uv run cityscapes-train-lightning` | Start Lightning training                       |

Use `--help` to inspect a command. Training commands support `--cfg job` to
display the configuration without training. Installed wheels contain the default
Hydra configurations and SQL files; dataset paths remain relative to the working
directory. Source defaults live in `src/cityscapes_tools/conf/`.

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

## Download dataset

Run the following commands from the `cityscapes-tools` directory. Check the
downloader's defaults without contacting Cityscapes or changing files:

```bash
uv run cityscapes-download --dry-run
```

Download the standard left images and fine annotations:

```bash
uv run cityscapes-download
```

This saves `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` under
`cityscapes-tools/data/cityscapes/` in your local repository. The downloader
verifies each archive's MD5 checksum; it does not extract the archives.

Pass other comma-separated package names from your Cityscapes download page when
needed:

```bash
uv run cityscapes-download \
  --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
  --destination=data/cityscapes
```

Use `--resume` to continue an interrupted download.
