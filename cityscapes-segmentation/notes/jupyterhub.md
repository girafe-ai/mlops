# JupyterHub

JupyterHub lets multiple users work with Jupyter notebooks through a shared server.
The demo setup lives in `jupyterhub-demo/`.

## Prerequisites

Install project dependencies:

```bash
uv sync
```

## Running

Start the hub from the project root:

```bash
uv run jupyterhub -f jupyterhub-demo/jupyterhub_config.py
```

The server starts at [http://localhost:8000](http://localhost:8000).
Log in with your system username and password.

## Stopping

Press `Ctrl+C` in the terminal where JupyterHub is running.

## Configuration

The config file is at `jupyterhub-demo/jupyterhub_config.py`.
It was generated with:

```bash
uv run jupyterhub --generate-config -f jupyterhub-demo/jupyterhub_config.py
```

All settings are commented out (defaults). To customize, uncomment and edit
the relevant lines. Common options:

```python
c.JupyterHub.port = 8000                    # change the port
c.JupyterHub.ip = '0.0.0.0'                 # listen on all interfaces
c.Spawner.default_url = '/lab'               # open JupyterLab instead of classic notebook
c.Authenticator.admin_users = {'your_user'}  # grant admin access
```

## File structure

```
jupyterhub-demo/
  jupyterhub_config.py      # server configuration
  jupyterhub_cookie_secret  # auto-generated secret (gitignored)
```

Runtime files (`*.pid`, `*.sqlite`, `jupyterhub_cookie_secret`) are
excluded from version control via `.gitignore`.
