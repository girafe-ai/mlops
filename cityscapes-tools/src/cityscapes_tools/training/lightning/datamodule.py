"""Lightning DataModule that exposes the DVC-generated Cityscapes subset."""

from pathlib import Path

import lightning
from torch.utils.data import DataLoader

from cityscapes_tools.data.cityscapes import CityscapesSegmentationDataset
from cityscapes_tools.data.loaders import create_dataloader, create_dataset


class CityscapesDataModule(lightning.LightningDataModule):
    """Create reproducible train, validation, test, and prediction loaders.

    DVC is responsible for obtaining the data artifact before this module is
    used. ``prepare_data`` only verifies that the local manifest is present;
    it never downloads data or stores distributed state.

    Args:
        manifest_path: DVC-generated filtered manifest.
        class_ids: Mapping from Cityscapes label IDs to training IDs.
        image_size: Output ``(height, width)`` for image/mask transforms.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker count.
        ignore_index: Configured value for ignored mask pixels and batch padding.
        test_split: Manifest split used for testing and prediction.
    """

    def __init__(
        self,
        manifest_path: Path,
        class_ids: dict[int, int],
        image_size: tuple[int, int],
        batch_size: int,
        num_workers: int,
        ignore_index: int,
        test_split: str = "val",
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.class_ids = class_ids
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.test_split = test_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Validate that DVC has made the filtered manifest available locally."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Filtered manifest is missing: {self.manifest_path}. Run `uv run dvc pull` first."
            )

    def setup(self, stage: str | None = None) -> None:
        """Create datasets required by the requested Trainer stage."""
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train", augment=True)
            self.val_dataset = self._dataset("val")
        if stage == "validate":
            self.val_dataset = self._dataset("val")
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training loader."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation loader."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the configured evaluation loader."""
        return self._loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Return the configured prediction loader."""
        return self._loader(self.test_dataset, shuffle=False)

    def _dataset(
        self,
        split: str,
        augment: bool = False,
    ) -> CityscapesSegmentationDataset:
        return create_dataset(
            manifest_path=self.manifest_path,
            split=split,
            class_ids=self.class_ids,
            ignore_index=self.ignore_index,
            image_size=self.image_size,
            augment=augment,
        )

    def _loader(
        self,
        dataset: CityscapesSegmentationDataset | None,
        shuffle: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a DataLoader.")
        return create_dataloader(
            dataset,
            ignore_index=self.ignore_index,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
