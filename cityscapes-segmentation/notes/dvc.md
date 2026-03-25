# DVC

Git tracks code and `.dvc` pointer files; DVC tracks the actual data in a
remote storage backend (S3, GCS, SSH, etc.).

## Setup

```bash
dvc init                              # initialise inside a Git repo
dvc remote add -d myremote s3://...   # register a default remote (cloud)
dvc remote add -d myremote /path/to/storage  # or use a local directory
```

## Local remote

A plain local directory can act as a DVC remote — useful when a cloud bucket
is not available or for sharing data on a shared filesystem / external drive.

```bash
dvc remote add -d local-data /Users/vl.naumov/Desktop/courses/dvc-data
```

This writes to `.dvc/config`:

```ini
[core]
    remote = local-data
['remote "local-data"']
    url = /Users/vl.naumov/Desktop/courses/dvc-data
```

`dvc push` / `dvc pull` then copy files to/from that directory instead of
uploading to S3. The directory structure mirrors the cache:
`<remote>/<md5[0:2]>/<md5[2:]>`.

This is how the remote is configured in this project.

## Tracking data

```bash
dvc add data/gtFine                   # hash + create data/gtFine.dvc
git add data/gtFine.dvc data/.gitignore
git commit -m "track dataset"
dvc push                              # upload to remote
```

## Getting data

```bash
dvc pull                              # download all tracked files
dvc pull data/gtFine.dvc             # download a single dataset
```

## Updating a dataset

```bash
# modify files in data/gtFine/, then:
dvc add data/gtFine
git add data/gtFine.dvc
git commit -m "dataset v2"
dvc push
```

## Switching versions

```bash
git checkout <commit> -- data/gtFine.dvc
dvc checkout data/gtFine.dvc          # restore that version from cache/remote
```

## Status

```bash
dvc status                            # workspace vs .dvc pointers
dvc status -c                         # local cache vs remote
```

## Cache

```bash
dvc gc -w                             # remove cache entries not used in current workspace
dvc gc -w --all-branches --all-tags   # keep everything referenced across all branches
```

## This project

| `.dvc` file | Content | Size |
|-------------|---------|------|
| `data/gtFine.dvc` | Ground-truth masks | ~772 MB, 20 000 files |
| `data/leftImg8bit.dvc` | RGB images | ~10.8 GB, 5 000 files |

Restore locally:

```bash
dvc pull                  # both datasets
dvc pull data/gtFine.dvc  # masks only (faster for dev)
```

## Config files

| File | Committed | Purpose |
|------|-----------|---------|
| `.dvc/config` | Yes | Remote URLs — shared |
| `.dvc/config.local` | No | Per-machine credentials |

```bash
# store credentials without committing them
dvc remote modify --local myremote access_key_id YOUR_KEY
dvc remote modify --local myremote secret_access_key YOUR_SECRET
```
