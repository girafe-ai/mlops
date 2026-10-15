import torch
from cityscapes_tools.training.lightning.module import CityscapesLightningModule
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader


def test_default_mlflow_logger_records_lightning_metrics_and_hyperparameters(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        config = compose(
            config_name="lightning",
            overrides=[
                "experiment.name=seminar",
                "experiment.run_id=demo",
            ],
        )

    logger = instantiate(
        config.logger,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    assert isinstance(logger, MLFlowLogger)
    model = CityscapesLightningModule(
        num_classes=3,
        base_channels=2,
        learning_rate=0.001,
        ignore_index=255,
    )
    batches = DataLoader(
        [
            {
                "images": torch.zeros(3, 8, 8),
                "masks": torch.zeros(8, 8, dtype=torch.long),
            }
        ],
        batch_size=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=batches, val_dataloaders=batches)
    trainer.test(model, dataloaders=batches, verbose=False)

    run = logger.experiment.get_run(logger.run_id)
    experiment = logger.experiment.get_experiment(run.info.experiment_id)
    assert experiment.name == "seminar"
    assert run.data.tags["mlflow.runName"] == "demo"
    assert run.info.status == "FINISHED"
    assert run.data.params["learning_rate"] == "0.001"
    assert run.data.params["ignore_index"] == "255"
    assert {
        "train_loss_step",
        "train_loss_epoch",
        "train_iou",
        "val_loss",
        "val_iou",
        "test_loss",
        "test_iou",
    } <= run.data.metrics.keys()
