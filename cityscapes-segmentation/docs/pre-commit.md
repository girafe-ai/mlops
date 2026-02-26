# Pre-commit

## Setup

Install dev dependencies (includes `pre-commit`):

```bash
uv sync --group dev
```

Install the git hook scripts into your local repo:

```bash
uv run pre-commit install
```

From this point on, hooks run automatically on every `git commit`.

## Running hooks manually

Run all hooks against all files:

```bash
uv run pre-commit run --all-files
```

Run a specific hook:

```bash
uv run pre-commit run strip-notebook-outputs --all-files
```

Run hooks only against staged files (same as what happens on commit):

```bash
uv run pre-commit run
```

## Configured hooks

The project's `.pre-commit-config.yaml` includes:

| Hook | Source | What it does |
|------|--------|--------------|
| `check-yaml` | [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) | Validates YAML syntax |
| `end-of-file-fixer` | [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) | Ensures files end with a newline |
| `trailing-whitespace` | [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) | Removes trailing whitespace |
| `check-added-large-files` | [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) | Blocks files larger than 500 KB |
| `ruff` | [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) | Lints Python code and sorts imports (`--select I`) |
| `ruff-format` | [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) | Formats Python code |
| `strip-notebook-outputs` | local (custom) | Strips outputs and execution counts from `.ipynb` files |

## Updating third-party hooks

```bash
uv run pre-commit autoupdate
```

## Custom hook: strip-notebook-outputs

This hook lives in the repository itself under `hooks/strip_notebook_outputs.py`.

### What it does

Before each commit, it scans every staged `.ipynb` file and:

1. Clears all cell `outputs` (the results of running cells)
2. Resets `execution_count` to `null`
3. Writes the cleaned notebook back to disk

If any file was modified, the hook **fails** the commit so you can review and re-stage the cleaned notebook.

### How it works

The hook is configured as a local hook in `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: strip-notebook-outputs
      name: Strip Jupyter notebook outputs
      entry: python cityscapes-segmentation/hooks/strip_notebook_outputs.py
      language: system
      types: [jupyter]
```

Each parameter explained:

- **`id`** — unique identifier, used to run just this hook: `pre-commit run strip-notebook-outputs`
- **`name`** — human-readable label shown in terminal output during the run
- **`entry`** — the command pre-commit executes. The path is relative to the
  **git root** (not the project subdirectory). Pre-commit appends matched
  filenames as arguments, so the actual call becomes:
  `python cityscapes-segmentation/hooks/strip_notebook_outputs.py path/to/notebook.ipynb`
- **`language: system`** — use the system Python directly, without creating an
  isolated virtualenv. This avoids installing all project dependencies just for
  a hook that only uses the standard library (`json`, `sys`)
- **`types: [jupyter]`** — file type filter. Only files identified as Jupyter
  notebooks (`.ipynb`) are passed to this hook; `.py`, `.md`, etc. are skipped

The hook must either:
- **Exit 0** — all clean, commit proceeds
- **Exit non-zero** or **modify files** — commit is blocked

### Typical commit flow

```bash
# You edit and run a notebook, then stage it
git add scripts/demo.ipynb

# Commit triggers the hook
git commit -m "update demo notebook"
# Output:
#   Strip Jupyter notebook outputs...Failed
#   - files were modified by this hook
#   Stripped outputs from scripts/demo.ipynb

# The notebook on disk is now clean — re-stage and commit again
git add scripts/demo.ipynb
git commit -m "update demo notebook"
# Output:
#   Strip Jupyter notebook outputs...Passed
```

### Creating your own custom hook

To add another local hook to this project:

1. Add a new module in `hooks/`, e.g. `hooks/my_new_hook.py`
2. Implement a `main()` function that reads filenames from `sys.argv[1:]` and returns `0` (pass) or `1` (fail)
3. Add the hook to `.pre-commit-config.yaml`:

```yaml
  - repo: local
    hooks:
      - id: my-new-hook
        name: My new hook
        entry: python cityscapes-segmentation/hooks/my_new_hook.py
        language: system
        types: [python]  # or [jupyter], [json], etc.
```

4. Test it:

```bash
uv run pre-commit run my-new-hook --all-files
```

### Important: save before committing

Pre-commit checks the **file on disk**, not what you see in the IDE.
If you run a notebook cell, you must **save the file** (Cmd+S) before
`git add` — otherwise the outputs exist only in IDE memory and the
hook won't see them.

### Project structure

```
.pre-commit-config.yaml      # which hooks to run (consumer)
.pre-commit-hooks.yaml       # which hooks this repo provides (publisher)
hooks/
  __init__.py
  strip_notebook_outputs.py  # the hook implementation
```

### `.pre-commit-config.yaml` vs `.pre-commit-hooks.yaml`

These two files look similar but serve opposite roles:

- **`.pre-commit-config.yaml`** — the **consumer** side. It declares which hooks
  this project runs on every commit. It references external repos
  (`pre-commit-hooks`, `ruff-pre-commit`) and local hooks.

- **`.pre-commit-hooks.yaml`** — the **publisher** side. It declares which hooks
  this repo **exposes to other projects**. If another team wants to reuse your
  `strip-notebook-outputs` hook, they can reference this repo in their own
  `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/yourname/your-repo
    rev: v0.1.0
    hooks:
      - id: strip-notebook-outputs
```

If you only use hooks locally and don't plan to share them,
`.pre-commit-hooks.yaml` is optional. It is included here for completeness
and in case the hook is reused across projects in the future.
