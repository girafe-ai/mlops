# What's New

Changes to Cityscapes tools, by date. Newest first.

## 2026-10-15

- Default Lightning logging to the built-in MLFlowLogger with a local server, optional checkpoint uploads, and a selectable CSV logger.
- Use data.ignore_index consistently for mask remapping, batch padding, loss, and Lightning IoU metrics in both training paths.
- Name experiment runs in Hydra and save best-validation-IoU and latest-epoch checkpoints under models, separately from CSV metrics.
- Log Lightning metrics and module hyperparameters locally with a Hydra-configured CSVLogger.
- Organize shared data/models and separate PyTorch/Lightning training packages; expose five installed CLI commands with packaged Hydra and SQL resources.
- Train a small Cityscapes segmentation model with either an explicit PyTorch loop or Lightning core objects.

## 2026-10-07

- Build reproducible Cityscapes metadata and filtered manifests with DVC, Hydra, and a local PostgreSQL database.

## 2026-10-01

- Find dataset information in the MkDocs guide; the separate `DATASET.md` file has been removed.

## 2026-09-30

- Browse a MkDocs guide for Cityscapes access and local downloads.
- Read a Python API reference generated from function and class docstrings.

## 2026-09-23

- Download Cityscapes archives to the local repository with dry-run, resume, and checksum verification options.
- Set up local Cityscapes credentials using `.env.example`.
- Find a dataset description and official links in the project documentation.
- Follow download instructions in `README.md`.
