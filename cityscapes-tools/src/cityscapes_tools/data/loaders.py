"""Dataset and DataLoader factories shared by raw PyTorch and Lightning."""

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cityscapes_tools.data.cityscapes import CityscapesSegmentationDataset
from cityscapes_tools.data.collate import pad_collate
from cityscapes_tools.data.transforms import SegmentationTransform


def create_dataset(
    manifest_path: Path,
    split: str,
    class_ids: dict[int, int],
    image_size: tuple[int, int],
    ignore_index: int,
    augment: bool = False,
) -> CityscapesSegmentationDataset:
    """Build a split using the configured mask ignore index and paired transforms."""
    return CityscapesSegmentationDataset(
        manifest_path=manifest_path,
        split=split,
        class_ids=class_ids,
        ignore_index=ignore_index,
        transform=SegmentationTransform(
            image_size=image_size,
            random_horizontal_flip=augment,
        ),
    )


def create_dataloader(
    dataset: CityscapesSegmentationDataset,
    batch_size: int,
    num_workers: int,
    ignore_index: int,
    shuffle: bool = False,
) -> DataLoader:
    """Build a loader whose mask padding uses the configured ignore index."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=partial(pad_collate, ignore_index=ignore_index),
    )
