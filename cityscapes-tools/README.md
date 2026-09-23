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
