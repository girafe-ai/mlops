"""Lightning orchestration for the Cityscapes segmentation course example."""

import lightning
import torch
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

from cityscapes_tools.models.tiny_unet import TinyUNet


class CityscapesLightningModule(lightning.LightningModule):
    """Train and evaluate :class:`TinyUNet` with Lightning orchestration.

    Args:
        num_classes: Number of classes predicted by the model.
        base_channels: Width of the first U-Net encoder block.
        learning_rate: AdamW learning rate.
        ignore_index: Mask value excluded from loss and IoU.
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        learning_rate: float = 0.001,
        *,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyUNet(num_classes=num_classes, base_channels=base_channels)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.validation_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-pixel class logits for an image batch."""
        return self.model(images)

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        """Compute and log one optimization loss."""
        return self._shared_step(batch, "train", self.train_iou)

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> None:
        """Compute and log validation loss and IoU."""
        self._shared_step(batch, "val", self.validation_iou)

    def test_step(self, batch: dict[str, object], batch_idx: int) -> None:
        """Compute and log test loss and IoU."""
        self._shared_step(batch, "test", self.test_iou)

    def predict_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, object]:
        """Return predicted masks together with the source image IDs."""
        logits = self(batch["images"])
        return {
            "image_ids": batch["image_ids"],
            "predictions": logits.argmax(dim=1),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW for the model's learnable parameters."""
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def _shared_step(
        self,
        batch: dict[str, object],
        stage: str,
        metric: MulticlassJaccardIndex,
    ) -> torch.Tensor:
        logits = self(batch["images"])
        masks = batch["masks"]
        loss = self.loss_function(logits, masks)
        predictions = logits.argmax(dim=1)
        metric(predictions, masks)
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}_iou", metric, on_step=False, on_epoch=True)
        return loss
