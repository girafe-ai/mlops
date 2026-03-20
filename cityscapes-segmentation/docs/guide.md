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

Edit the constants at the top of `cityscapes-segmentation/train.py` to match
your setup, then run:

```bash
cd cityscapes-segmentation
uv run python train.py
```

Key constants:

| Constant | Default | Description |
|---|---|---|
| `DATA_ROOT` | `../data` | Path to the data directory |
| `EPOCHS` | `50` | Number of training epochs |
| `BATCH_SIZE` | `4` | Samples per batch |
| `LR` | `1e-4` | Initial learning rate |
| `ENCODER` | `resnet34` | SMP encoder backbone |
| `MAX_SAMPLES` | `None` | Limit samples per split (useful for quick runs) |

Checkpoints are saved to `checkpoints/best.pth` (best val mIoU) and
`checkpoints/last.pth` (latest epoch).

To resume a training run:

```python
RESUME = "checkpoints/last.pth"
```

## Inference

Edit the constants at the top of `cityscapes-segmentation/infer.py`, then run:

```bash
cd cityscapes-segmentation
uv run python infer.py
```

Key constants:

| Constant | Default | Description |
|---|---|---|
| `CHECKPOINT` | `checkpoints/best.pth` | Trained model checkpoint |
| `INPUT` | `../data/leftImg8bit/val` | Image file or directory |
| `OUTPUT_DIR` | `None` | Save results here; `None` saves next to source |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |

Output files are saved as `<original_name>_pred.png` with Cityscapes class colours.
