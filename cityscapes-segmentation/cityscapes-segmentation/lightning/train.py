"""Train a UNet segmentation model on Cityscapes using PyTorch Lightning + Hydra.

Usage:

    uv run python train.py
    uv run python train.py training.epochs=10 data.batch_size=8
    uv run python train.py training.resume=checkpoints/last.ckpt
"""

import hydra
import pytorch_lightning as pl
from data import CityscapesDataModule
from model import SegmentationModel
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the segmentation model using config from Hydra.

    Args:
        cfg: Hydra config composed from conf/config.yaml and its defaults.
    """
    pl.seed_everything(cfg.training.seed, workers=True)

    datamodule = CityscapesDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        height=cfg.data.height,
        width=cfg.data.width,
        max_samples=cfg.data.max_samples,
    )

    model = SegmentationModel(
        encoder=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="best",
            monitor="val_miou",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=False,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume,
    )


if __name__ == "__main__":
    train()
