# Cityscapes Segmentation

Semantic segmentation of urban driving scenes using a UNet model trained on the
[Cityscapes](https://www.cityscapes-dataset.com/) dataset.

## Overview

The project ships two implementations under `cityscapes-segmentation/`:

### `raw/` — plain PyTorch
- **`data.py`** — dataset loading and preprocessing from local Cityscapes files
- **`train.py`** — training loop with AMP, checkpointing and validation
- **`infer.py`** — inference on single images or directories, with colourised output
- **`utils.py`** — shared transforms and mIoU metric

### `lightning/` — PyTorch Lightning + Hydra
- **`model.py`** — `SegmentationModel` (`LightningModule`) and training logic
- **`data.py`** — `CityscapesDataModule` (`LightningDataModule`) for data loading
- **`train.py`** — `Trainer`-based training script driven by Hydra config
- **`infer.py`** — inference script driven by Hydra config
- **`utils.py`** — shared transforms and mIoU metric

All hyperparameters for the Lightning variant are managed through `conf/` (see the [Guide](guide.md)).

## Requirements

- Python ≥ 3.10
- A Cityscapes account to download the dataset (see the [Guide](guide.md))
- CUDA-capable GPU recommended for training

## Quick links

- [Guide](guide.md) — installation, data download, training and inference walkthrough
- [API Reference](api.md) — full module and function documentation
