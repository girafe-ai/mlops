"""Train a UNet segmentation model on the Cityscapes dataset.

Usage:

    uv run python train.py
"""

import logging
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from data import NUM_CLASSES, get_dataloader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import compute_miou

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_ROOT = "../../data"
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-4
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 4
HEIGHT = 512
WIDTH = 1024
RESUME: str | None = None
MAX_SAMPLES: int | None = None  # e.g. 50 to use only 50 images per split
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run a single training epoch with mixed-precision forward and backward passes.

    Iterates over all batches in ``loader``, computes the loss, backpropagates
    with AMP gradient scaling, and updates the model weights.

    Args:
        model: The segmentation model to train.
        loader: DataLoader for the training split.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        optimizer: Parameter optimizer.
        device: Device to run computation on.
        scaler: AMP gradient scaler; scaling is disabled automatically on CPU.
        epoch: Current epoch number (1-indexed), used for the progress bar label.
        total_epochs: Total number of epochs, used for the progress bar label.

    Returns:
        Mean training loss over all samples in the epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Evaluate the model on a validation split without gradient computation.

    Runs inference over all batches in ``loader``, accumulates the loss, and
    computes mean IoU across the full split via a confusion matrix.

    Args:
        model: The segmentation model to evaluate.
        loader: DataLoader for the validation split.
        criterion: Loss function used for the validation loss.
        device: Device to run computation on.
        epoch: Current epoch number (1-indexed), used for the progress bar label.
        total_epochs: Total number of epochs, used for the progress bar label.

    Returns:
        Tuple of ``(val_loss, val_miou)`` — mean loss per sample and mean IoU
        over the 19 Cityscapes training classes.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = total_loss / len(loader.dataset)
    miou = compute_miou(torch.cat(all_preds), torch.cat(all_targets))
    return val_loss, miou


def train(
    data_root: str = "../data",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 4,
    height: int = 512,
    width: int = 1024,
    resume: str | None = None,
    max_samples: int | None = 50,
) -> None:
    """Train a UNet model on Cityscapes.

    Args:
        data_root: Path to the root data directory (must contain
            ``leftImg8bit/`` and ``gtFine/`` sub-directories).
        epochs: Total number of training epochs.
        batch_size: Samples per batch.
        lr: Initial learning rate for AdamW.
        encoder: SMP encoder backbone name (e.g. ``"resnet34"``).
        encoder_weights: Pre-trained weights for the encoder.
        checkpoint_dir: Directory to save model checkpoints.
        num_workers: DataLoader worker processes.
        height: Input image resize height.
        width: Input image resize width.
        resume: Path to a checkpoint to resume training from.
        max_samples: Cap each split at this many samples. ``None`` uses all.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading datasets from %s", data_root)
    train_loader = get_dataloader(
        data_root,
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        height=height,
        width=width,
        pin_memory=device.type == "cuda",
        max_samples=max_samples,
    )
    val_loader = get_dataloader(
        data_root,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        height=height,
        width=width,
        pin_memory=device.type == "cuda",
        max_samples=max_samples,
    )
    log.info(
        "Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader)
    )

    log.info("Building model: UNet + %s", encoder)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_miou = 0.0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("val_miou", 0.0)
        log.info(
            "Resumed from epoch %d, best mIoU so far: %.4f", start_epoch, best_miou
        )

    epoch_bar = tqdm(range(start_epoch, epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.perf_counter()
        ep_label = epoch + 1

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ep_label, epochs
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion, device, ep_label, epochs
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            mIoU=f"{val_miou:.4f}",
        )
        log.info(
            "Epoch %03d/%d  train_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  (%.1fs)",
            ep_label,
            epochs,
            train_loss,
            val_loss,
            val_miou,
            elapsed,
        )

        save_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_miou": val_miou,
            "val_loss": val_loss,
        }

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(save_payload, checkpoint_path / "best.pth")
            log.info(
                "New best mIoU %.4f — checkpoint saved to %s/best.pth",
                best_miou,
                checkpoint_dir,
            )

        torch.save(save_payload, checkpoint_path / "last.pth")

    log.info("Training complete. Best val mIoU: %.4f", best_miou)


if __name__ == "__main__":
    train(
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        encoder=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        checkpoint_dir=CHECKPOINT_DIR,
        num_workers=NUM_WORKERS,
        height=HEIGHT,
        width=WIDTH,
        resume=RESUME,
        max_samples=MAX_SAMPLES,
    )
