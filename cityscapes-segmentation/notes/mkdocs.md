# MkDocs

MkDocs is a static site generator for project documentation written in Markdown.
`mkdocs-material` is the most popular theme; `mkdocstrings` auto-generates API
reference pages from Python docstrings.

---

## Setup for a new project

### 1. Add dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.6.1",
    "mkdocs-material>=9.7.5",
    "mkdocstrings[python]>=1.0.3",
]
```

Install:

```bash
uv sync --extra docs
```

### 2. Initialise MkDocs

```bash
uv run mkdocs new .
```

This creates:

```
docs/
    index.md
mkdocs.yml
```

### 3. Configure `mkdocs.yml`

Minimal configuration with Material theme, search, and docstring generation:

```yaml
site_name: My Project

theme:
  name: material

markdown_extensions:
  - toc:
      toc_depth: 3       # how deep the TOC sidebar goes (h1 → h3)

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths:
            - src         # directory that contains your Python package
          options:
            show_source: false
            docstring_style: google   # google | numpy | sphinx
```

### 4. Create `docs/api.md`

Reference individual modules or members using the `:::` directive:

```markdown
# API Reference

## Data

::: mypackage.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - MyClass
        - my_function
```

- `show_root_heading: false` — suppresses the auto-generated module heading,
  preventing duplicate TOC entries alongside the `##` Markdown heading.
- `show_root_toc_entry: false` — removes the module name from the TOC.
- `heading_level: 3` — renders each member as `###`, nested under the `##`
  section heading in the sidebar.

---

## Serving and building

```bash
# live preview with hot reload
uv run --extra docs mkdocs serve

# build static site into site/
uv run --extra docs mkdocs build

# deploy to GitHub Pages
uv run --extra docs mkdocs gh-deploy
```

---

## Themes

| Value | Package | Notes |
|---|---|---|
| `material` | `mkdocs-material` | Most popular; supports dark mode, admonitions, tabs |
| `readthedocs` | built-in | Classic ReadTheDocs look |
| `mkdocs` | built-in | Default minimal theme |

Switch theme in `mkdocs.yml`:

```yaml
theme:
  name: material
```

Material-specific extras (optional):

```yaml
theme:
  name: material
  palette:
    - scheme: default      # light mode
    - scheme: slate        # dark mode
  features:
    - navigation.tabs
    - navigation.top
    - content.code.copy
```

---

## Docstring styles

| Value | Format | Common in |
|---|---|---|
| `google` | `Args:` / `Returns:` sections | Google, most ML projects |
| `numpy` | `Parameters\n----------` underline style | NumPy, SciPy |
| `sphinx` | `:param x:` / `:returns:` inline tags | Traditional Python docs |

**Google style example:**

```python
def train(epochs: int = 50, lr: float = 1e-4) -> None:
    """Train the model.

    Args:
        epochs: Number of training epochs.
        lr: Initial learning rate.
    """
```

**NumPy style example:**

```python
def train(epochs: int = 50, lr: float = 1e-4) -> None:
    """Train the model.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate.
    """
```

---

## Useful `mkdocstrings` options

| Option | Default | Description |
|---|---|---|
| `show_source` | `true` | Show the source code below each member |
| `show_root_heading` | `true` | Render the module/class name as a heading |
| `show_root_toc_entry` | `true` | Add module/class to the TOC |
| `heading_level` | `2` | Heading level (`#`) used for each member |
| `docstring_style` | `google` | Docstring format to parse |
| `members` | all | Explicit list of members to document |
| `inherited_members` | `false` | Include inherited methods |
| `show_signature` | `true` | Show function/method signature |
