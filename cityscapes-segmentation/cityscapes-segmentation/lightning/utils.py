"""Shared utility functions."""

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 19


def build_transforms(split: str, height: int = 512, width: int = 1024) -> A.Compose:
    """Build albumentations transform pipeline for a given split.

    Args:
        split: One of ``"train"``, ``"val"``, or ``"test"``.
            Horizontal flip augmentation is added for ``"train"`` only.
        height: Resize height in pixels.
        width: Resize width in pixels.

    Returns:
        Configured :class:`albumentations.Compose` pipeline.
    """
    shared = [A.Resize(height, width)]
    augment = [A.HorizontalFlip(p=0.5)] if split == "train" else []
    normalize = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(shared + augment + normalize)


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU over all valid classes.

    Args:
        preds: Predicted class indices, shape ``[B, H, W]``.
        targets: Ground-truth class indices, shape ``[B, H, W]``.
        num_classes: Number of valid classes.
        ignore_index: Label value to exclude from the metric.

    Returns:
        Mean IoU across classes that appear in the batch.
    """
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=preds.device
    )

    valid = targets != ignore_index
    p = preds[valid]
    t = targets[valid]

    indices = num_classes * t + p
    confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )

    tp = confusion.diagonal()
    fn = confusion.sum(dim=1) - tp
    fp = confusion.sum(dim=0) - tp
    denom = tp + fp + fn
    iou = torch.where(
        denom > 0, tp.float() / denom.float(), torch.zeros_like(tp, dtype=torch.float)
    )
    return iou[denom > 0].mean().item()
