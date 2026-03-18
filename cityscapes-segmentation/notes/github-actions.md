# GitHub Actions

GitHub Actions run automated checks on every push and pull request.
Workflow files live in `.github/workflows/` at the **git root** (`mlops/`).

## Existing workflows

### pre-commit (`.github/workflows/pre-commit.yaml`)

Runs all pre-commit hooks on every PR and push to `main`/`master`.
If any hook fails, the PR check is marked as failed.

```yaml
name: pre-commit

on:
  pull_request:
    branches: [26s-mipt]
  push:
    branches: [26s-mipt]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: cityscapes-segmentation
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - uses: pre-commit/action@v3.0.1
        with:
          extra_args: --config cityscapes-segmentation/.pre-commit-config.yaml --all-files
```

## Adding a new workflow

1. Create a YAML file in `.github/workflows/` (at the git root):

```
mlops/
  .github/
    workflows/
      pre-commit.yaml    # existing
      my-new-action.yaml # new
  cityscapes-segmentation/
    ...
```

2. Define the trigger, job, and steps. Minimal template:

```yaml
name: my-new-action

on:
  pull_request:
    branches: [main, master]

jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt
        working-directory: cityscapes-segmentation

      - name: Run my check
        run: python my_script.py
        working-directory: cityscapes-segmentation
```

3. Commit and push — GitHub picks up any `.yaml` file in `.github/workflows/`
   automatically.

## Key concepts

### Triggers (`on`)

| Trigger | When it runs |
|---------|-------------|
| `pull_request` | When a PR is opened or updated |
| `push` | When commits are pushed to the branch |
| `schedule` | On a cron schedule (e.g. nightly) |
| `workflow_dispatch` | Manually from the GitHub UI |

### Working directory

Since the git root is `mlops/` but the project is in `cityscapes-segmentation/`,
use `working-directory` to run commands in the right folder:

```yaml
defaults:
  run:
    working-directory: cityscapes-segmentation
```

Or per-step:

```yaml
- name: Run tests
  run: pytest
  working-directory: cityscapes-segmentation
```

### Using third-party actions

Actions from the marketplace are referenced as `uses: owner/repo@version`:

```yaml
- uses: actions/checkout@v4        # check out the repo
- uses: actions/setup-python@v5    # install Python
- uses: pre-commit/action@v3.0.1   # run pre-commit
```

Pin to a specific version (`@v4`, `@v3.0.1`) for reproducibility.

### Viewing results

After pushing, go to the **Actions** tab in your GitHub repository to see
workflow runs, logs, and failure details.
