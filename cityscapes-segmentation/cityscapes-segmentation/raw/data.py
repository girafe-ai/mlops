"""Cityscapes dataset loading and preprocessing.

Provides :class:`CityscapesDataset` and the :func:`get_dataloader` factory
for loading locally stored Cityscapes images and ground-truth masks.

Expected data layout::

    data_root/
        leftImg8bit/{train,val,test}/{city}/*_leftImg8bit.png
        gtFine/{train,val,test}/{city}/*_gtFine_labelIds.png

Download the dataset via ``scripts/download.py`` if not already present.
"""

from pathlib import Path

import albumentations as A
import numpy as np
from cityscapesscripts.helpers.labels import labels as cs_labels
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import NUM_CLASSES, build_transforms

# Map raw label IDs to 19 train IDs; 255 = ignore
ID_TO_TRAIN_ID = {
    label.id: (label.trainId if label.trainId not in (-1, 255) else 255)
    for label in cs_labels
}
ID_TO_TRAIN_ID_ARRAY = np.full(256, 255, dtype=np.uint8)
for raw_id, train_id in ID_TO_TRAIN_ID.items():
    if 0 <= raw_id < 256:
        ID_TO_TRAIN_ID_ARRAY[raw_id] = train_id


class CityscapesDataset(Dataset):
    """Cityscapes semantic segmentation dataset loaded from local files.

    Expected directory layout::

        data_root/
            leftImg8bit/{split}/{city}/*_leftImg8bit.png
            gtFine/{split}/{city}/*_gtFine_labelIds.png

    Args:
        data_root: Path to the root data directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        transforms: Optional albumentations ``Compose`` transform.
            Defaults to resize + normalize (+ horizontal flip for train).
        height: Resize height in pixels.
        width: Resize width in pixels.
        max_samples: Cap the dataset at this many samples. ``None`` uses all.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        transforms: A.Compose | None = None,
        height: int = 512,
        width: int = 1024,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        self.split = split
        self.transforms = transforms or build_transforms(split, height, width)

        img_dir = data_root / "leftImg8bit" / split
        mask_dir = data_root / "gtFine" / split

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                "Download leftImg8bit_trainvaltest.zip via scripts/download.py"
            )
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        mask_files = {}
        for mask_path in sorted(mask_dir.rglob("*_gtFine_labelIds.png")):
            stem = "_".join(mask_path.stem.split("_")[:3])  # city_seq_frame
            mask_files[stem] = mask_path

        self.samples = []
        for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
            stem = "_".join(img_path.stem.split("_")[:3])
            if stem in mask_files:
                self.samples.append((img_path, mask_files[stem]))

        if not self.samples:
            raise RuntimeError(
                f"No matched image–mask pairs found in {data_root} for split '{split}'"
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        """Return the number of image–mask pairs in the split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Load, preprocess and return one sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary with keys:

            - ``"image"``: ``FloatTensor[3, H, W]`` — normalised RGB image.
            - ``"mask"``: ``LongTensor[H, W]`` — per-pixel train IDs
                (0–18); unlabelled pixels are set to ``255``.
        """
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_raw = np.array(Image.open(mask_path), dtype=np.uint8)

        mask = ID_TO_TRAIN_ID_ARRAY[mask_raw]

        transformed = self.transforms(image=image, mask=mask)
        return {
            "image": transformed["image"].float(),
            "mask": transformed["mask"].long(),
        }


def get_dataloader(
    data_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    height: int,
    width: int,
    pin_memory: bool,
    max_samples: int | None,
) -> DataLoader:
    """Return a DataLoader for the given Cityscapes split.

    Args:
        data_root: Path to the root data directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        batch_size: Number of samples per batch.
        num_workers: Worker processes for data loading.
        height: Resize height in pixels.
        width: Resize width in pixels.
        pin_memory: Pin tensors to CUDA page-locked memory.
        max_samples: Cap the dataset at this many samples. ``None`` uses all.

    Returns:
        Configured :class:`torch.utils.data.DataLoader`.
    """
    dataset = CityscapesDataset(
        data_root=data_root,
        split=split,
        height=height,
        width=width,
        max_samples=max_samples,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
