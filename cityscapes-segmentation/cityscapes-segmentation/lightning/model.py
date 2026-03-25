"""PyTorch Lightning module for Cityscapes segmentation."""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import NUM_CLASSES, compute_miou


class SegmentationModel(pl.LightningModule):
    """UNet segmentation model wrapped as a LightningModule.

    Args:
        encoder: SMP encoder backbone name (e.g. ``"resnet34"``).
        encoder_weights: Pre-trained weights for the encoder.
        lr: Initial learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        epochs: Total training epochs, used for CosineAnnealingLR T_max.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        encoder_weights: str = "imagenet",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["mask"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["mask"])
        self._val_preds.append(logits.argmax(dim=1).cpu())
        self._val_targets.append(batch["mask"].cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        miou = compute_miou(preds, targets)
        self.log("val_miou", miou, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
