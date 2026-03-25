"""Cityscapes dataset and LightningDataModule.

Expected data layout::

    data_root/
        leftImg8bit/{train,val,test}/{city}/*_leftImg8bit.png
        gtFine/{train,val,test}/{city}/*_gtFine_labelIds.png
"""

from pathlib import Path

import albumentations as A
import numpy as np
import pytorch_lightning as pl
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

    Args:
        data_root: Path to the root data directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        transforms: Optional albumentations ``Compose`` transform.
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
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_raw = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = ID_TO_TRAIN_ID_ARRAY[mask_raw]

        transformed = self.transforms(image=image, mask=mask)
        return {
            "image": transformed["image"].float(),
            "mask": transformed["mask"].long(),
        }


class CityscapesDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping the Cityscapes dataset.

    Args:
        data_root: Path to the root data directory.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        height: Resize height in pixels.
        width: Resize width in pixels.
        max_samples: Cap each split at this many samples. ``None`` uses all.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        height: int,
        width: int,
        max_samples: int | None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = CityscapesDataset(
                self.hparams.data_root,
                "train",
                height=self.hparams.height,
                width=self.hparams.width,
                max_samples=self.hparams.max_samples,
            )
            self.val_ds = CityscapesDataset(
                self.hparams.data_root,
                "val",
                height=self.hparams.height,
                width=self.hparams.width,
                max_samples=self.hparams.max_samples,
            )
        if stage in ("test", None):
            self.test_ds = CityscapesDataset(
                self.hparams.data_root,
                "test",
                height=self.hparams.height,
                width=self.hparams.width,
                max_samples=self.hparams.max_samples,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
