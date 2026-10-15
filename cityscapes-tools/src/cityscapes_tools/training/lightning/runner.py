"""Assemble Lightning core objects and run the Trainer."""

from pathlib import Path

import lightning
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import DictConfig

from cityscapes_tools.training.lightning.datamodule import CityscapesDataModule
from cityscapes_tools.training.lightning.module import CityscapesLightningModule


def run(config: DictConfig) -> None:
    """Run a named experiment with a configured logger and best/latest checkpoints."""
    class_ids = {int(key): int(value) for key, value in config.data.class_ids.items()}
    data_module = CityscapesDataModule(
        manifest_path=Path(config.data.manifest_path),
        class_ids=class_ids,
        image_size=(config.data.image_height, config.data.image_width),
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        ignore_index=config.data.ignore_index,
    )
    model = CityscapesLightningModule(
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
        learning_rate=config.training.learning_rate,
        ignore_index=config.data.ignore_index,
    )
    best_checkpoint = instantiate(config.checkpoint.best)
    latest_checkpoint = instantiate(config.checkpoint.latest)
    trainer = lightning.Trainer(
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=config.training.max_epochs,
        logger=instantiate(config.logger),
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=[
            best_checkpoint,
            latest_checkpoint,
            EarlyStopping(
                monitor=best_checkpoint.monitor,
                mode=best_checkpoint.mode,
                patience=config.training.patience,
            ),
        ],
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(
        model, datamodule=data_module, ckpt_path=best_checkpoint.best_model_path
    )
