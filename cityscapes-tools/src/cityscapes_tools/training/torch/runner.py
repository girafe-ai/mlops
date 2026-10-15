"""Assemble shared data and model components for raw PyTorch training."""

from pathlib import Path

import torch
from omegaconf import DictConfig

from cityscapes_tools.data.loaders import create_dataloader, create_dataset
from cityscapes_tools.models.tiny_unet import TinyUNet
from cityscapes_tools.training.torch.loop import (
    resolve_device,
    train_one_epoch,
    validate,
)


def run(config: DictConfig) -> None:
    """Build data/model objects and run the explicit training loop."""
    class_ids = {int(key): int(value) for key, value in config.data.class_ids.items()}
    train_dataset = create_dataset(
        manifest_path=Path(config.data.manifest_path),
        split="train",
        class_ids=class_ids,
        ignore_index=config.data.ignore_index,
        image_size=(config.data.image_height, config.data.image_width),
        augment=True,
    )
    validation_dataset = create_dataset(
        manifest_path=Path(config.data.manifest_path),
        split="val",
        class_ids=class_ids,
        ignore_index=config.data.ignore_index,
        image_size=(config.data.image_height, config.data.image_width),
    )
    train_loader = create_dataloader(
        train_dataset,
        ignore_index=config.data.ignore_index,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        ignore_index=config.data.ignore_index,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    device = resolve_device(config.training.device)
    model = TinyUNet(
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)

    for epoch in range(config.training.max_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        validation_loss = validate(
            model,
            validation_loader,
            loss_function,
            device,
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={validation_loss:.4f}"
        )
