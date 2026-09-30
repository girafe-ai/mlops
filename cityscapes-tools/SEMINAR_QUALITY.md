# Project Quality Automation

This seminar turns a Python package into a project that can check itself locally
and in CI.

## 1. Pre-commit

`pre-commit` runs fast checks before a commit is created. Keep its configuration
in the Git repository root and install it through the project's `uv` environment.

From the repository root:

```bash
uv add --directory cityscapes-tools --dev pre-commit ruff
uv run --directory cityscapes-tools pre-commit run --all-files --config ../.pre-commit-config.yaml
uv run --directory cityscapes-tools pre-commit install --config ../.pre-commit-config.yaml
```

Recommended hooks:

- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, and
  `check-merge-conflict` from `pre-commit-hooks`;
- `check-added-large-files` to protect the repository from accidental binaries;
- Gitleaks to detect secrets;
- Ruff for formatting, linting, and import ordering;
- a local `verify-documented-commands` hook that runs explicitly marked, safe
  commands from the documentation.

For this new project, use Ruff as the single Python quality tool:

| Need         | Ruff command                  | Traditional alternative |
| ------------ | ----------------------------- | ----------------------- |
| Format code  | `ruff format`                 | Black                   |
| Lint code    | `ruff check`                  | Flake8                  |
| Sort imports | `ruff check --select I --fix` | isort                   |

Do not run Ruff together with Black, isort, and Flake8: they duplicate work and
can disagree about style. The traditional trio is useful to explain the separate
roles of formatter, import sorter, and linter; Ruff is the recommended setup.

```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Do not run slow tests or real downloads as pre-commit hooks.

The local hook should validate a documentation contract: commands marked with
`<!-- verify-command -->` in `README.md` or `docs/*.md` must still run. It must
only execute an explicit allowlist of safe commands, such as a downloader
`--dry-run`; never execute arbitrary Markdown as shell code.

## 2. GitHub Actions

For pull requests targeting `26f_mipt`, run the same checks on a clean runner:

```yaml
on:
  pull_request:
    branches: [26f_mipt]
```

```text
checkout -> install uv -> uv sync -> pre-commit run --all-files
         -> pytest -> mkdocs build --strict
```

The workflow is an independent safety net: contributors can skip a local hook,
but they cannot bypass the PR check.

## 3. Changelog

Keep `cityscapes-tools/CHANGELOG.md` in the Keep a Changelog style:

```md
## [Unreleased]

### Added

### Changed

### Fixed
```

Update it **before the commit**, in the same commit as a user-visible change.
Do not add entries for formatting, tests, CI, lockfile-only changes, or internal
refactors with no user-visible effect.

Put this rule in the repository-root `AGENTS.md` so the agent reviews the final
diff and updates the `Unreleased` section without being reminded. The agent may
draft the entry; a human reviews it and decides whether it belongs in the
changelog.

```md
When modifying Cityscapes package code or its CLI, review the completed diff.
For a user-visible change, add one concise English bullet to CHANGELOG.md under
Unreleased. Skip tests, CI, formatting, lockfile-only changes, and internal
refactors. Do not change versions, commit, or push unless explicitly asked.
```

## 4. Documentation with MkDocs

Use a Markdown-first stack:

```text
MkDocs + Material for MkDocs + mkdocstrings[python]
```

```bash
uv add --directory cityscapes-tools --dev mkdocs-material "mkdocstrings[python]"
uv run --directory cityscapes-tools mkdocs serve
uv run --directory cityscapes-tools mkdocs build --strict
```

Suggested structure:

```text
mkdocs.yml
docs/
  index.md
  usage.md
  api.md
```

`mkdocstrings` can render the downloader API from docstrings with:

```md
::: cityscapes_tools.downloader
```

Useful commands:

```bash
uv run mkdocs serve
uv run mkdocs build --strict
```

Mark a small number of safe usage commands in the documentation. The local
pre-commit hook verifies that those commands stay runnable.

## 5. Pytest

Keep normal tests offline and deterministic. The initial suite can contain five
examples:

| Test class  | Test name                                    | Purpose                                                            |
| ----------- | -------------------------------------------- | ------------------------------------------------------------------ |
| Smoke       | `test_downloader_import_has_no_side_effects` | Importing the module creates no HTTP session or files.             |
| Unit        | `test_parse_packages_rejects_paths`          | Invalid package names are rejected.                                |
| Unit        | `test_get_credentials_prefers_environment`   | Environment values take precedence over `.env`.                    |
| Integration | `test_download_writes_verified_archive`      | A fake HTTP session exercises login, chunks, and MD5 verification. |
| Contract    | `test_documented_dry_run_succeeds`           | A documented safe command remains runnable.                        |

Use `parametrize` for input variants, `tmp_path` for temporary files,
`monkeypatch` for environment variables and network boundaries, `capsys` for
CLI output, and `pytest.raises` for errors. A real Cityscapes download is an
opt-in end-to-end test, never part of the default CI suite.

## 6. Logging instead of diagnostic `print`

Use `print()` only for intentional command output that a user should see.
Use the standard `logging` module for diagnostics:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Downloading %s", package_name)
logger.warning("Checksum mismatch for %s", package_name)
```

Logging provides levels, timestamps, configurable destinations, and useful CI
output. Never log passwords, API tokens, or the contents of `.env`.
