"""Explicit PyTorch training and validation loops for the segmentation example."""

import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Run one standard PyTorch optimization epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images, masks = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Run an evaluation epoch and return mean validation loss."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        images, masks = _move_batch_to_device(batch, device)
        total_loss += loss_function(model(images), masks).item()
    return total_loss / len(loader)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to MPS, CUDA, or CPU in that priority order."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_batch_to_device(
    batch: dict[str, object], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the tensor fields produced by ``pad_collate`` to a device."""
    return batch["images"].to(device), batch["masks"].to(device)
