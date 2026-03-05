# Docker

## Core concepts

**Image** -- a read-only template with a filesystem snapshot and metadata
(base OS, installed packages, application code, default command).
Images are built from a `Dockerfile` and stored in registries (Docker Hub, GHCR, etc.).

**Container** -- a running (or stopped) instance of an image.
Containers share the host kernel but have their own isolated filesystem,
process tree, and network stack.

**Volume** -- a persistent storage mechanism managed by Docker.
Unlike bind mounts, volumes are independent of the host directory layout
and survive container removal.

**Network** -- a virtual network that lets containers communicate.
By default every container gets its own network namespace;
user-defined networks enable DNS-based service discovery between containers.

### Architecture

```text
┌────────────────────────────────────────────┐
│               Docker CLI                   │
│            (docker build/run/…)            │
└──────────────────┬─────────────────────────┘
                   │  REST API
┌──────────────────▼─────────────────────────┐
│            Docker Daemon (dockerd)         │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Images  │ │Containers│ │  Volumes   │  │
│  └──────────┘ └──────────┘ └────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │            containerd / runc         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

---

## CLI commands reference

### Image management

| Command | Description | Example |
|---------|-------------|---------|
| `docker build` | Build image from Dockerfile | `docker build -t myapp .` |
| `docker pull` | Download image from registry | `docker pull python:3.10-slim` |
| `docker push` | Upload image to registry | `docker push myuser/myapp:v1` |
| `docker images` | List local images | `docker images` |
| `docker rmi` | Remove image | `docker rmi myapp:latest` |
| `docker tag` | Create an alias for an image | `docker tag myapp:latest myapp:v1` |
| `docker history` | Show layers of an image | `docker history python:3.10-slim` |
| `docker inspect` | Show detailed image/container JSON | `docker inspect myapp:latest` |

### Container lifecycle

| Command | Description | Example |
|---------|-------------|---------|
| `docker run` | Create and start a container | `docker run --rm myapp` |
| `docker start` | Start a stopped container | `docker start my_ctr` |
| `docker stop` | Gracefully stop a container (SIGTERM) | `docker stop my_ctr` |
| `docker kill` | Force stop (SIGKILL) | `docker kill my_ctr` |
| `docker restart` | Stop then start | `docker restart my_ctr` |
| `docker rm` | Remove a stopped container | `docker rm my_ctr` |
| `docker ps` | List running containers | `docker ps -a` |
| `docker logs` | Show container stdout/stderr | `docker logs -f my_ctr` |
| `docker stats` | Live resource usage | `docker stats` |

### Interaction

| Command | Description | Example |
|---------|-------------|---------|
| `docker exec` | Run a command inside a running container | `docker exec -it my_ctr bash` |
| `docker attach` | Attach to the main process stdin/stdout | `docker attach my_ctr` |
| `docker cp` | Copy files between host and container | `docker cp my_ctr:/app/out.txt .` |

### Volumes

| Command | Description |
|---------|-------------|
| `docker volume create` | Create a named volume |
| `docker volume ls` | List volumes |
| `docker volume inspect` | Show volume details |
| `docker volume rm` | Remove a volume |
| `docker volume prune` | Remove all unused volumes |

### Networks

| Command | Description |
|---------|-------------|
| `docker network create` | Create a user-defined network |
| `docker network ls` | List networks |
| `docker network inspect` | Show network details |
| `docker network connect` | Connect a container to a network |
| `docker network disconnect` | Disconnect a container from a network |
| `docker network rm` | Remove a network |

### System

| Command | Description |
|---------|-------------|
| `docker system df` | Show disk usage by images, containers, volumes |
| `docker system prune` | Remove all stopped containers, unused networks, dangling images |
| `docker system prune -a` | Same as above plus all unused images |
| `docker info` | System-wide Docker information |

---

## `docker run` flags

`docker run` is the most flag-rich command. Here are the essential flags:

| Flag | Short | Description | Example |
|------|-------|-------------|---------|
| `--name` | | Assign a name to the container | `docker run --name train myapp` |
| `--rm` | | Automatically remove container on exit | `docker run --rm myapp` |
| `--detach` | `-d` | Run in background | `docker run -d myapp` |
| `--interactive` | `-i` | Keep STDIN open | `docker run -i myapp` |
| `--tty` | `-t` | Allocate a pseudo-terminal | `docker run -it myapp bash` |
| `--env` | `-e` | Set environment variable | `docker run -e LR=0.01 myapp` |
| `--env-file` | | Load env vars from file | `docker run --env-file .env myapp` |
| `--volume` | `-v` | Bind mount or named volume | `docker run -v $(pwd)/out:/outputs myapp` |
| `--mount` | | More explicit mount syntax | `docker run --mount type=bind,src=./out,dst=/outputs myapp` |
| `--publish` | `-p` | Map host port to container port | `docker run -p 8080:80 nginx` |
| `--network` | | Connect to a Docker network | `docker run --network my_net myapp` |
| `--workdir` | `-w` | Override working directory | `docker run -w /src myapp` |
| `--user` | `-u` | Run as specific UID:GID | `docker run -u 1000:1000 myapp` |
| `--gpus` | | GPU access (requires nvidia-docker) | `docker run --gpus all nvidia/cuda` |
| `--memory` | `-m` | Memory limit | `docker run -m 2g myapp` |
| `--cpus` | | CPU limit | `docker run --cpus 2.0 myapp` |
| `--entrypoint` | | Override ENTRYPOINT | `docker run --entrypoint bash myapp` |
| `--platform` | | Target platform (cross-arch) | `docker run --platform linux/amd64 myapp` |
| `--restart` | | Restart policy | `docker run --restart unless-stopped myapp` |
| `--hostname` | `-h` | Set container hostname | `docker run -h train-box myapp` |
| `--read-only` | | Mount root FS as read-only | `docker run --read-only myapp` |
| `--tmpfs` | | Mount a tmpfs | `docker run --tmpfs /tmp myapp` |
| `--init` | | Run tini as PID 1 | `docker run --init myapp` |
| `--privileged` | | Full host capabilities | `docker run --privileged myapp` |

---

## `docker exec` flags

| Flag | Short | Description | Example |
|------|-------|-------------|---------|
| `--interactive` | `-i` | Keep STDIN open even if not attached | `docker exec -i my_ctr python script.py < input.txt` |
| `--tty` | `-t` | Allocate a pseudo-TTY | `docker exec -t my_ctr ls --color` |
| `--detach` | `-d` | Run command in background | `docker exec -d my_ctr python long_job.py` |
| `--env` | `-e` | Set an environment variable | `docker exec -e DEBUG=1 my_ctr python app.py` |
| `--env-file` | | Read env vars from a file | `docker exec --env-file .env my_ctr python app.py` |
| `--workdir` | `-w` | Working directory inside container | `docker exec -w /app/src my_ctr python main.py` |
| `--user` | `-u` | Run as specific user (name or UID:GID) | `docker exec -u nobody my_ctr whoami` |
| `--privileged` | | Give extended Linux capabilities | `docker exec --privileged my_ctr mount /dev/sda1 /mnt` |
| `--detach-keys` | | Override the key sequence to detach | `docker exec --detach-keys="ctrl-x" -it my_ctr bash` |

### Common `docker exec` patterns

```bash
# open an interactive shell inside a running container
docker exec -it my_ctr bash

# run a one-off command and see the output
docker exec my_ctr cat /app/metrics.json

# run a background process inside the container
docker exec -d my_ctr python /app/evaluate.py

# run as root even if the container runs as non-root user
docker exec -u 0 my_ctr apt-get update

# pipe data into a container process
echo '{"key": "value"}' | docker exec -i my_ctr python -c "import sys, json; print(json.load(sys.stdin))"

# set working directory for the command
docker exec -w /app/data my_ctr ls -la
```

---

## Dockerfile instructions

| Instruction | Description | Example |
|-------------|-------------|---------|
| `FROM` | Base image | `FROM python:3.10-slim` |
| `RUN` | Execute command during build | `RUN pip install -r requirements.txt` |
| `COPY` | Copy files from build context | `COPY train.py .` |
| `ADD` | Like COPY but supports URLs and auto-extract tar | `ADD data.tar.gz /data` |
| `WORKDIR` | Set working directory for subsequent instructions | `WORKDIR /app` |
| `ENV` | Set environment variable (persists at runtime) | `ENV PYTHONUNBUFFERED=1` |
| `ARG` | Build-time variable (not available at runtime) | `ARG PYTHON_VERSION=3.10` |
| `EXPOSE` | Document which port the app listens on | `EXPOSE 8080` |
| `CMD` | Default command (overridden by `docker run` args) | `CMD ["python", "train.py"]` |
| `ENTRYPOINT` | Fixed command prefix (args appended) | `ENTRYPOINT ["python"]` |
| `VOLUME` | Create a mount point | `VOLUME /data` |
| `USER` | Switch to non-root user | `USER appuser` |
| `LABEL` | Add metadata | `LABEL version="1.0"` |
| `HEALTHCHECK` | Container health probe | `HEALTHCHECK CMD curl -f http://localhost/` |
| `SHELL` | Override default shell | `SHELL ["/bin/bash", "-c"]` |

### `CMD` vs `ENTRYPOINT`

- **`CMD`** -- provides a default command and arguments. Entirely replaced
  when you pass arguments to `docker run`.
- **`ENTRYPOINT`** -- sets the fixed executable. Arguments from `docker run`
  are appended to it.
- Common pattern: use both together:

```dockerfile
ENTRYPOINT ["python"]
CMD ["train.py"]
```

Then `docker run myapp` runs `python train.py`, and
`docker run myapp evaluate.py` runs `python evaluate.py`.

---

## `.dockerignore`

Works like `.gitignore` -- excludes files from the build context sent to the
Docker daemon. Reduces build time and prevents leaking secrets.

Example:

```text
__pycache__/
*.pyc
.git/
.venv/
outputs/
*.egg-info/
.env
```

---

## Hands-on: CatBoost training in a container

The `docker-seminar/` directory (at the repo root) contains a ready-to-use
example: train a CatBoost model on the California Housing dataset, evaluate it,
and save the model, metrics, and plots.

### Directory layout

```text
docker-seminar/
├── train.py              # shared training script
├── requirements.txt      # shared Python dependencies
├── .dockerignore         # shared build exclusions
├── plain/                # plain Docker (no orchestration)
│   ├── Dockerfile
│   └── Dockerfile.multistage
└── compose/              # Docker Compose variant
    └── docker-compose.yaml
```

Shared files (`train.py`, `requirements.txt`, `.dockerignore`) live at the root
of `docker-seminar/`. The `plain/` and `compose/` subdirectories contain only
what is unique to each approach.

---

### Part A -- Plain Docker (`plain/`)

#### Step 1 -- Build the image

```bash
cd docker-seminar
docker build -t catboost-train -f plain/Dockerfile .
```

The `-f` flag points to the Dockerfile inside `plain/`, while `.` sets the
build context to `docker-seminar/` (where `train.py` and `requirements.txt`
live).

#### Step 2 -- Run training (default hyperparameters)

```bash
docker run --rm -v $(pwd)/outputs:/outputs catboost-train
```

After it finishes, check the `outputs/` directory:

```bash
ls outputs/
# model.cbm  metrics.json  predictions_vs_actual.png  feature_importance.png  residuals.png
```

```bash
cat outputs/metrics.json
```

#### Step 3 -- Run with custom hyperparameters

Pass hyperparameters via environment variables to practice the `-e` flag:

```bash
docker run --rm \
  -e LEARNING_RATE=0.05 \
  -e DEPTH=8 \
  -e ITERATIONS=1000 \
  -v $(pwd)/outputs:/outputs \
  catboost-train
```

Compare `metrics.json` with the previous run.

#### Step 4 -- Inspect a running container

Start training in detached mode so you can interact with the container:

```bash
docker run -d --name train_run \
  -v $(pwd)/outputs:/outputs \
  catboost-train
```

While it runs:

```bash
# check logs in real time
docker logs -f train_run

# open a shell inside the container
docker exec -it train_run bash

# check what's in /outputs from inside the container
docker exec train_run ls -la /outputs

# see resource usage
docker stats train_run --no-stream
```

Clean up:

```bash
docker rm train_run
```

#### Step 5 -- Copy files out of a container

Instead of using a volume, you can copy files after the run:

```bash
docker run --name train_copy catboost-train

docker cp train_copy:/outputs/metrics.json ./metrics.json
docker cp train_copy:/outputs/predictions_vs_actual.png ./predictions_vs_actual.png

docker rm train_copy
```

#### Step 6 -- Resource limits

Run with memory and CPU constraints:

```bash
docker run --rm \
  -m 1g \
  --cpus 1.0 \
  -v $(pwd)/outputs:/outputs \
  catboost-train
```

Check resource consumption from another terminal:

```bash
docker stats
```

---

### Part B -- Multi-stage builds (`plain/Dockerfile.multistage`)

A multi-stage build uses multiple `FROM` instructions. The final image only
contains what is copied from earlier stages, which can reduce size
significantly -- depending on what the builder stage installs.

The key idea: use a **heavy** base image (with compilers, headers, build tools)
in the builder stage, and a **slim** base image in the runtime stage. Only the
installed packages are copied over -- the build tools stay behind.

#### `Dockerfile.multistage`

```dockerfile
# --- stage 1: build dependencies in the FULL image (has gcc, headers, etc.) ---
FROM python:3.10 AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- stage 2: runtime uses the SLIM image (no compilers, no pip cache) ---
FROM python:3.10-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]
```

Notice the difference in base images:
- **Builder:** `python:3.10` (~920 MB) -- includes gcc, make, libc headers, etc.
- **Runtime:** `python:3.10-slim` (~150 MB) -- minimal Debian with just Python.

The builder stage can compile C extensions from source if needed. The runtime
stage only receives the final installed packages via the venv, without any of
the build toolchain.

#### When multi-stage helps

Multi-stage builds save space when the builder stage contains tools that are
not needed at runtime (compilers, `-dev` packages, build caches). If all your
dependencies install as pre-built wheels (like catboost), the size difference
may be small because there is nothing to leave behind. The pattern pays off
most when you have packages that compile from source.

#### Build and compare sizes

```bash
cd docker-seminar

# single-stage (slim base, no build tools)
docker build -t catboost-train:single -f plain/Dockerfile .

# multi-stage (full base for build, slim for runtime)
docker build -t catboost-train:multi -f plain/Dockerfile.multistage .

# compare
docker images | grep catboost-train
```

You will notice that the `multi` image is **not smaller** (and may even be
slightly larger) than `single`. This is expected: catboost, scikit-learn, and
matplotlib all install as pre-built wheels -- no compilation happens, so there
are no build tools to discard. The venv structure itself adds a small overhead.

This is an important takeaway: multi-stage builds are not a universal
optimization. They help when the builder stage pulls in heavy tools (gcc,
`-dev` packages, Rust toolchain) that are only needed during `pip install`.
For pure wheel installs, a simple single-stage slim image is the better choice.

---

### Part C -- Docker Compose (`compose/`)

Docker Compose lets you define multi-container setups (or multiple
configurations of the same container) in a single YAML file. Here we use it
to launch three training runs with different hyperparameters in parallel.

#### `docker-compose.yaml`

```yaml
services:
  train-default:
    build:
      context: ..
      dockerfile: plain/Dockerfile
    volumes:
      - ./outputs/default:/outputs
    environment:
      - LEARNING_RATE=0.1
      - DEPTH=6
      - ITERATIONS=500

  train-deep:
    build:
      context: ..
      dockerfile: plain/Dockerfile
    volumes:
      - ./outputs/deep:/outputs
    environment:
      - LEARNING_RATE=0.05
      - DEPTH=10
      - ITERATIONS=1000

  train-fast:
    build:
      context: ..
      dockerfile: plain/Dockerfile
    volumes:
      - ./outputs/fast:/outputs
    environment:
      - LEARNING_RATE=0.3
      - DEPTH=4
      - ITERATIONS=200
```

Key points:
- **`context: ..`** -- the build context is the parent `docker-seminar/`
  directory where `train.py` and `requirements.txt` live.
- **`dockerfile: plain/Dockerfile`** -- reuses the same Dockerfile from
  `plain/`, no duplication.
- Each service writes to its own output subdirectory
  (`outputs/default/`, `outputs/deep/`, `outputs/fast/`).

#### Step 1 -- Run all three experiments

```bash
cd docker-seminar/compose
docker compose up --build
```

This builds the image once (shared across services) and starts all three
training runs. Each one writes its model, metrics, and plots to a separate
output directory.

#### Step 2 -- Run a single service

```bash
docker compose up --build train-deep
```

#### Step 3 -- Compare results

```bash
cat outputs/default/metrics.json
cat outputs/deep/metrics.json
cat outputs/fast/metrics.json
```

#### Step 4 -- Run in background and check logs

```bash
docker compose up --build -d
docker compose logs -f train-deep
```

#### Step 5 -- Clean up

```bash
docker compose down
```

---

## Useful one-liners

```bash
# remove all stopped containers
docker container prune

# remove all unused images
docker image prune -a

# remove everything (containers, images, volumes, networks)
docker system prune -a --volumes

# show container IP address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my_ctr

# export container filesystem as tar
docker export my_ctr > backup.tar

# show real-time events from the daemon
docker events
```
