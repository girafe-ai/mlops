import csv
from pathlib import Path
from unittest.mock import Mock

from cityscapes_tools.training.lightning import runner
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf


def test_csv_logger_writes_metrics_and_hyperparameters(tmp_path) -> None:
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        config = compose(
            config_name="lightning",
            overrides=[
                "logger=csv",
                "experiment.name=seminar",
                "experiment.run_id=demo",
            ],
        )

    assert "logger" in config
    config.logger.save_dir = str(tmp_path)
    logger = instantiate(config.logger)
    assert isinstance(logger, CSVLogger)

    logger.log_hyperparams({"learning_rate": 0.001})
    logger.log_metrics({"train_loss": 0.5, "val_iou": 0.25}, step=0)
    logger.save()

    log_dir = Path(logger.log_dir)
    with (log_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["train_loss"] == "0.5"
    assert rows[0]["val_iou"] == "0.25"
    assert OmegaConf.load(log_dir / "hparams.yaml").learning_rate == 0.001
    assert log_dir == tmp_path / "seminar" / "demo"


def test_runner_passes_configured_logger_to_trainer(monkeypatch, tmp_path) -> None:
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        config = compose(config_name="lightning")
    OmegaConf.set_struct(config, False)
    config.logger = OmegaConf.create(
        {
            "_target_": "lightning.pytorch.loggers.CSVLogger",
            "save_dir": str(tmp_path),
            "name": "seminar",
        }
    )
    config.training.log_every_n_steps = 1
    config.data.ignore_index = -1
    trainer_factory = Mock()
    monkeypatch.setattr(runner.lightning, "Trainer", trainer_factory)

    runner.run(config)

    arguments = trainer_factory.call_args.kwargs
    assert isinstance(arguments["logger"], CSVLogger)
    assert arguments["logger"].name == "seminar"
    assert arguments["log_every_n_steps"] == 1
    checkpoints = [
        callback
        for callback in arguments["callbacks"]
        if isinstance(callback, ModelCheckpoint)
    ]
    assert len(checkpoints) == 2
    best, latest = checkpoints
    assert best.monitor == "val_iou"
    assert best.mode == "max"
    assert best.filename == "best"
    assert best.save_top_k == 1
    assert latest.monitor is None
    assert latest.save_top_k == 1
    assert latest.filename == "epoch_{epoch:03d}"
    early_stopping = next(
        callback
        for callback in arguments["callbacks"]
        if isinstance(callback, EarlyStopping)
    )
    assert early_stopping.monitor == best.monitor
    assert early_stopping.mode == best.mode
    trainer_factory.return_value.fit.assert_called_once()
    trainer_factory.return_value.test.assert_called_once()
    model = trainer_factory.return_value.fit.call_args.args[0]
    data_module = trainer_factory.return_value.fit.call_args.kwargs["datamodule"]
    assert data_module.ignore_index == config.data.ignore_index
    assert model.loss_function.ignore_index == config.data.ignore_index
    for metric in (model.train_iou, model.validation_iou, model.test_iou):
        assert metric.ignore_index == config.data.ignore_index


def test_checkpoint_names_and_directories_follow_experiment_config(tmp_path) -> None:
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        config = compose(config_name="lightning")
    assert "experiment" in config
    assert "checkpoint" in config
    config.experiment.name = "seminar"
    config.experiment.run_id = "demo"
    config.checkpoint.save_dir = str(tmp_path / "models")

    best = instantiate(config.checkpoint.best)
    latest = instantiate(config.checkpoint.latest)
    expected_dir = tmp_path / "models" / "seminar" / "demo"
    assert Path(best.dirpath) == expected_dir
    assert Path(latest.dirpath) == expected_dir
    assert Path(best.format_checkpoint_name({"epoch": 3})).name == "best.ckpt"
    assert Path(latest.format_checkpoint_name({"epoch": 3})).name == "epoch_003.ckpt"
    assert best.save_weights_only is False
    assert latest.save_weights_only is False
    assert best._enable_version_counter is False
    assert latest._save_on_train_epoch_end is True
