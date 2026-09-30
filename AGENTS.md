# Course repository instructions

## Scope

- Apply Cityscapes-specific rules only when working in `cityscapes-tools/`.
- Keep course-facing documentation and changelog entries in English.
- Do not read, print, commit, or modify `.env` files or credentials unless the
  user explicitly asks.

## Cityscapes tools

- Target Python 3.10 and keep the code compatible with that version.
- Do not add `from __future__ import annotations` unless it solves a concrete
  forward-reference or annotation-evaluation problem. Explain the reason in the
  change summary when it is necessary.
- Use `uv` for Python commands and dependency changes. Do not edit dependency
  declarations or `uv.lock` by hand.
- Use `uv add <package>` for runtime dependencies, `uv add --dev <package>`
  for development dependencies, and `uv remove <package>` to remove one.
- After a dependency change, ensure both `pyproject.toml` and `uv.lock` are
  included in the change. Run project commands with `uv run <command>`.
- Do not commit `.venv`; it is a local, reproducible environment created by
  `uv sync`.
- Keep implementation code in `cityscapes-tools/src/cityscapes_tools/`; keep
  scripts as thin command-line wrappers.
- Do not make network requests or download Cityscapes data unless explicitly
  requested.

## Changelog

When modifying Cityscapes package code, its CLI, or user-facing documentation:

1. Review the completed diff before reporting the task as finished.
2. If the change is user-visible, update
   `cityscapes-tools/CHANGELOG.md` under `## [Unreleased]`.
3. Add one concise English bullet under `Added`, `Changed`, or `Fixed`.
4. Skip changelog entries for tests, CI, formatting, lockfile-only updates, and
   internal refactors with no user-visible effect.
5. Do not change the package version, create a release, commit, or push unless
   the user explicitly asks.

## Verification

- Run the smallest relevant local verification after a change.
- Once available, run the project's configured pre-commit hooks, pytest suite,
  and MkDocs build when they are relevant to the files changed.
