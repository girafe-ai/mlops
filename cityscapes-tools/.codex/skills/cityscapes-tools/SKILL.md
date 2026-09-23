---
name: cityscapes-tools
description: Work on the cityscapes-tools Python project, including its uv dependency workflow and project-specific Python conventions.
---

# cityscapes-tools

Use this skill for changes to this repository. Keep its project configuration and dependency lockfile managed with `uv`.

## Python

- The project targets Python 3.10; keep code compatible with that version.
- Do not add `from __future__ import annotations` unless a concrete annotation-evaluation or forward-reference problem requires it. If it is necessary, explain why in the change summary.
- This section is the home for additional project-specific Python conventions as they are adopted.

## Dependencies

Use `uv` rather than editing dependency declarations or the lockfile by hand.

- Add a runtime dependency with `uv add <package>`.
- Add a development-only dependency with `uv add --dev <package>`.
- Remove a dependency with `uv remove <package>`.
- After dependency changes, commit both `pyproject.toml` and `uv.lock`.
- Run project commands in the managed environment with `uv run <command>`; for example, `uv run python -m cityscapes_tools`.

## Getting started from GitHub

After cloning the repository, users need `uv` and a compatible Python interpreter. From the repository root, run:

```bash
uv sync
```

This creates or updates `.venv` and installs the exact locked dependency set. Then run project commands through `uv run`, such as:

```bash
uv run cityscapes-tools
```

When dependencies change upstream, users should pull the latest commit and run `uv sync` again. Do not commit `.venv`; it is a local, reproducible environment.
