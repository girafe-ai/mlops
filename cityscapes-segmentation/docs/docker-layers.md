# Docker Layers

## What is a layer

A Docker image is not a single file -- it is a stack of **layers**. Each layer
is a read-only filesystem diff: it records what files were added, modified, or
deleted compared to the layer below it.

Layers are created by specific Dockerfile instructions:

| Creates a layer | Metadata only (no layer) |
|-----------------|--------------------------|
| `FROM` | `ENV` |
| `RUN` | `ARG` |
| `COPY` | `WORKDIR` |
| `ADD` | `CMD` |
| | `ENTRYPOINT` |
| | `EXPOSE` |
| | `LABEL` |
| | `USER` |
| | `SHELL` |
| | `HEALTHCHECK` |

Metadata-only instructions update the image configuration (default command,
environment variables, working directory, etc.) but do not add a filesystem
layer. They show up as 0 B in `docker history`.

When you run a container, Docker adds a thin **writable layer** on top of the
image layers. All writes inside the container go to this layer. When the
container is removed, the writable layer is deleted -- the image layers remain
untouched.

---

## Layer stack

Consider this Dockerfile (from `docker-seminar/plain/Dockerfile`):

```dockerfile
FROM python:3.10-slim          # layer 1 (base image, multiple sub-layers)
WORKDIR /app                   # metadata only
COPY requirements.txt .        # layer 2
RUN pip install ... -r req...  # layer 3
COPY train.py .                # layer 4
CMD ["python", "train.py"]     # metadata only
```

The resulting image looks like this:

```text
┌──────────────────────────────────────────┐
│         Writable container layer         │  ← exists only at runtime
├──────────────────────────────────────────┤
│  Layer 4: COPY train.py           ~4 KB  │
├──────────────────────────────────────────┤
│  Layer 3: RUN pip install       ~1.1 GB  │
├──────────────────────────────────────────┤
│  Layer 2: COPY requirements.txt   ~30 B  │
├──────────────────────────────────────────┤
│                                          │
│  Layer 1: FROM python:3.10-slim          │
│  (base image -- multiple sub-layers:     │
│   Debian slim + Python interpreter)      │
│                                       │
│                                ~150 MB   │
└──────────────────────────────────────────┘
```

Docker uses a **union filesystem** (typically overlay2 on Linux) to merge all
these read-only layers into a single coherent filesystem view. When a file
exists in multiple layers, the topmost version wins.

---

## Layer caching

When you run `docker build`, Docker checks each instruction from top to bottom:

1. If the instruction and its inputs have not changed since the last build,
   Docker reuses the cached layer (`CACHED` or `Using cache` in the output).
2. Once a cache miss occurs, **all subsequent layers are rebuilt** -- even if
   their instructions haven't changed.

This is why instruction order in a Dockerfile matters:

```dockerfile
# GOOD: requirements.txt changes rarely, train.py changes often
COPY requirements.txt .
RUN pip install -r requirements.txt   # cached most of the time
COPY train.py .                       # only this layer rebuilds

# BAD: copying everything first invalidates pip cache on any code change
COPY . .
RUN pip install -r requirements.txt   # rebuilds every time train.py changes
```

### What invalidates the cache

| Instruction | Cache invalidated when... |
|-------------|---------------------------|
| `RUN` | The command string changes |
| `COPY` / `ADD` | Any source file content or metadata changes |
| `FROM` | The base image tag resolves to a different digest |

---

## Inspecting layers with CLI

### `docker history`

Shows each layer, the instruction that created it, and its size:

```bash
docker history catboost-train:single
```

Example output:

```text
IMAGE          CREATED        CREATED BY                                      SIZE
14879dde1e76   2 hours ago    CMD ["python" "train.py"]                       0B
<missing>      2 hours ago    COPY train.py . # buildkit                      4.1kB
<missing>      2 hours ago    RUN pip install --no-cache-dir -r requi...       1.1GB
<missing>      2 hours ago    COPY requirements.txt . # buildkit              30B
<missing>      2 hours ago    WORKDIR /app                                    0B
<missing>      3 weeks ago    ...python:3.10-slim base layers...              150MB
```

To see full (untruncated) commands:

```bash
docker history --no-trunc catboost-train:single
```

### `docker inspect`

Shows the layer digests (SHA256 hashes) that make up the image:

```bash
docker inspect catboost-train:single | jq '.[0].RootFS'
```

Example output:

```json
{
  "Type": "layers",
  "Layers": [
    "sha256:aabb1122...",
    "sha256:ccdd3344...",
    "sha256:eeff5566...",
    "sha256:00112233..."
  ]
}
```

The number of entries in `Layers` matches the number of non-zero-size rows in
`docker history`.

### `docker save` + tar extraction

You can export the image to a tar archive and look at the raw layer files:

```bash
docker save catboost-train:single -o image.tar
mkdir image-contents
tar -xf image.tar -C image-contents
ls image-contents/
```

Modern Docker uses the **OCI image format**. Inside you will find:

```text
image-contents/
├── blobs/
│   └── sha256/
│       ├── <hash1>     ← gzip-compressed layer tar
│       ├── <hash2>     ← JSON metadata (config, manifest, index)
│       ├── ...
│       └── <hashN>     ← gzip-compressed layer tar
├── index.json          ← entry point, points to the manifest
├── manifest.json       ← lists config blob + layer blobs in order
└── oci-layout          ← format version marker
```

**Important:** not every blob is a layer. The `blobs/sha256/` directory is a
content-addressed store that mixes layer tarballs, the image config JSON, and
OCI index/manifest JSON files. Only the blobs listed in the `"Layers"` array
of `manifest.json` are actual layers. The rest are JSON metadata. If you try
to `tar -tzf` a JSON blob, you'll get `Unrecognized archive format`.

Read `manifest.json` to see which blobs are layers and which is the config:

```bash
cat image-contents/manifest.json | python3 -m json.tool
```

The output has two key fields:
- `"Config"` -- path to the image config JSON blob
- `"Layers"` -- ordered list of layer blobs (bottom to top)

Each **layer** blob is a gzip-compressed tar. Use `tar -tzf` (with `z`) to
list its contents:

```bash
tar -tzf image-contents/blobs/sha256/<layer-hash> | head -20
```

Clean up when done:

```bash
rm -rf image.tar image-contents
```

### `dive` (third-party tool)

[dive](https://github.com/wagoodman/dive) is an interactive TUI that lets you
browse layers, see which files each layer added/modified/deleted, and spot
wasted space.

```bash
# install (macOS)
brew install dive

# run
dive catboost-train:single
```

---

## Hands-on exercises

These exercises use the existing `catboost-train:single` image and a
purpose-built `layers/Dockerfile` from `docker-seminar/`.

### Exercise 1 -- Inspect layers of the training image

```bash
docker history catboost-train:single
```

Questions:
- Which layer is the largest? What instruction created it?
- Which layers are 0 B? Why?
- How many filesystem layers does the image have?

Verify the count with:

```bash
docker inspect catboost-train:single | jq '.[0].RootFS.Layers | length'
```

### Exercise 2 -- Build the layers demo image

The `docker-seminar/layers/Dockerfile` is designed to create easily
identifiable layers with varying sizes.

```bash
cd docker-seminar
docker build -t layers-demo -f layers/Dockerfile .
```

Then inspect:

```bash
docker history layers-demo
```

Identify:
- The layer created by `apt-get install curl`
- The layer created by `pip install`
- The 10 MB dummy layer created by `dd`
- The 0 B metadata-only rows (`WORKDIR`, `CMD`)

### Exercise 3 -- Look inside the layers

Export the image and extract the archive:

```bash
docker save layers-demo -o layers-demo.tar
mkdir layers-unpacked
tar -xf layers-demo.tar -C layers-unpacked
ls layers-unpacked/
```

You'll see the OCI layout: `blobs/`, `index.json`, `manifest.json`,
`oci-layout`. First, read `manifest.json` to see the ordered list of layer
blobs:

```bash
cat layers-unpacked/manifest.json | python3 -m json.tool
```

The `"Layers"` array lists blob paths from bottom (base image) to top (last
instruction). Each blob is a gzip-compressed tar.

List the size and contents of each layer blob:

```bash
python3 -c "
import json
m = json.load(open('layers-unpacked/manifest.json'))
for layer in m[0]['Layers']:
    print(layer)
" | while read blob; do
  size=$(ls -lh "layers-unpacked/$blob" | awk '{print $5}')
  echo "=== $size  $blob ==="
  tar -tzf "layers-unpacked/$blob" | head -5
  echo "..."
done
```

Find the layer that contains the 10 MB dummy file:

```bash
python3 -c "
import json
m = json.load(open('layers-unpacked/manifest.json'))
for layer in m[0]['Layers']:
    print(layer)
" | while read blob; do
  if tar -tzf "layers-unpacked/$blob" 2>/dev/null | grep -q "dummy.bin"; then
    echo "Found dummy.bin in: $blob"
    tar -tzf "layers-unpacked/$blob"
  fi
done
```

Clean up:

```bash
rm -rf layers-demo.tar layers-unpacked
```

### Exercise 4 -- Observe cache invalidation

**Step 1.** Rebuild without changes -- everything is cached:

```bash
docker build -t layers-demo -f layers/Dockerfile .
```

Every step should show `CACHED`.

**Step 2.** Touch `train.py` to change its modification time:

```bash
touch train.py
docker build -t layers-demo -f layers/Dockerfile .
```

Observe: `COPY train.py .` and all layers **after** it are rebuilt. Layers
**before** it (apt-get, pip install) are still cached.

**Step 3.** Now edit `requirements.txt` (e.g., add a comment line):

```bash
echo "# trigger rebuild" >> requirements.txt
docker build -t layers-demo -f layers/Dockerfile .
```

Observe: `COPY requirements.txt .` invalidates the cache, and `pip install`
is rebuilt too -- even though the actual packages haven't changed. This
demonstrates the cascading invalidation rule.

Revert the change:

```bash
git checkout requirements.txt
```

### Exercise 5 -- Why instruction order matters

Create a bad Dockerfile that copies everything before installing packages:

```bash
cat > /tmp/Dockerfile.bad << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "train.py"]
EOF
```

Build it, then touch `train.py` and rebuild:

```bash
docker build -t bad-order -f /tmp/Dockerfile.bad .
touch train.py
docker build -t bad-order -f /tmp/Dockerfile.bad .
```

Notice that `pip install` is rebuilt even though only `train.py` changed.
Compare with the original Dockerfile where `COPY requirements.txt` is
separate from `COPY train.py` -- pip install stays cached when only code
changes.

Clean up:

```bash
docker rmi bad-order layers-demo
rm /tmp/Dockerfile.bad
```
