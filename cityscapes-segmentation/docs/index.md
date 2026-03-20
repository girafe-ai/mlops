# Cityscapes Segmentation

Semantic segmentation of urban driving scenes using a UNet model trained on the
[Cityscapes](https://www.cityscapes-dataset.com/) dataset.

## Overview

The project is structured around three main modules:

- **`data.py`** — dataset loading and preprocessing from local Cityscapes files
- **`train.py`** — full training loop with checkpointing and validation
- **`infer.py`** — inference on single images or directories, with colourised output

Shared helpers (transforms, mIoU metric) live in **`utils.py`**.

## Requirements

- Python ≥ 3.10
- A Cityscapes account to download the dataset (see the [Guide](guide.md))
- CUDA-capable GPU recommended for training

## Quick links

- [Guide](guide.md) — installation, data download, training and inference walkthrough
- [API Reference](api.md) — full module and function documentation
