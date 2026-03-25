# Guide

## Installation

Install the project with [uv](https://github.com/astral-sh/uv):

```bash
git clone <repo-url>
cd cityscapes-segmentation
uv sync
```

To also install documentation dependencies:

```bash
uv sync --extra docs
```

## Downloading the dataset

You need a free [Cityscapes account](https://www.cityscapes-dataset.com/register/).

Download ground-truth masks (required for training):

```bash
uv run python scripts/download.py download \
    --package_names '["gtFine_trainvaltest.zip"]' \
    --destination_path ./data
```

Download RGB images (required for training, ~11 GB):

```bash
uv run python scripts/download.py download \
    --package_names '["leftImg8bit_trainvaltest.zip"]' \
    --destination_path ./data
```

After extraction the data layout should be:

```
data/
    gtFine/{train,val,test}/{city}/*_gtFine_labelIds.png
    leftImg8bit/{train,val,test}/{city}/*_leftImg8bit.png
```

## Training

### raw/

Edit the constants at the top of `cityscapes-segmentation/raw/train.py` to
match your setup, then run:

```bash
cd cityscapes-segmentation/raw
uv run python train.py
```

Key constants:

| Constant | Default | Description |
|---|---|---|
| `DATA_ROOT` | `../../data` | Path to the data directory |
| `EPOCHS` | `50` | Number of training epochs |
| `BATCH_SIZE` | `4` | Samples per batch |
| `LR` | `1e-4` | Initial learning rate |
| `ENCODER` | `resnet34` | SMP encoder backbone |
| `MAX_SAMPLES` | `None` | Limit samples per split (useful for quick runs) |

Checkpoints are saved to `checkpoints/best.pth` (best val mIoU) and
`checkpoints/last.pth` (latest epoch).

To resume a training run set `RESUME = "checkpoints/last.pth"` in the file.

### lightning/

All hyperparameters are controlled via `conf/`. Run with defaults:

```bash
cd cityscapes-segmentation/lightning
uv run python train.py
```

Override any value on the command line:

```bash
uv run python train.py training.epochs=10 data.batch_size=8 data.max_samples=50
```

Key config files and their parameters:

| File | Key parameters |
|---|---|
| `conf/data/default.yaml` | `data_root`, `batch_size`, `num_workers`, `height`, `width`, `max_samples` |
| `conf/model/default.yaml` | `encoder`, `encoder_weights` |
| `conf/training/default.yaml` | `epochs`, `lr`, `weight_decay`, `precision`, `checkpoint_dir`, `resume` |

Checkpoints are saved to `checkpoints/best.ckpt` (best val mIoU) and
`checkpoints/last.ckpt` (latest epoch).

To resume a training run:

```bash
uv run python train.py training.resume=checkpoints/last.ckpt
```

## Inference

### raw/

Edit the constants at the top of `cityscapes-segmentation/raw/infer.py`, then run:

```bash
cd cityscapes-segmentation/raw
uv run python infer.py
```

Key constants:

| Constant | Default | Description |
|---|---|---|
| `CHECKPOINT` | `checkpoints/best.pth` | Trained model checkpoint |
| `INPUT` | `../../data/leftImg8bit/val` | Image file or directory |
| `OUTPUT_DIR` | `None` | Save results here; `None` saves next to source |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |

### lightning/

```bash
cd cityscapes-segmentation/lightning
uv run python infer.py
```

Override via CLI:

```bash
uv run python infer.py inference.checkpoint=checkpoints/best.ckpt inference.output_dir=predictions
```

Key config parameters (`conf/inference/default.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | `checkpoints/best.ckpt` | Trained model checkpoint |
| `input` | `../../data/leftImg8bit/val` | Image file or directory |
| `output_dir` | `null` | Save results here; `null` saves next to source |
| `device` | `auto` | `auto`, `cpu`, or `cuda` |

Output files are saved as `<original_name>_pred.png` with Cityscapes class colours.
