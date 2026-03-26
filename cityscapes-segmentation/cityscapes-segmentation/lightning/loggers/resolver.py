"""Resolve the configured logger from Hydra config."""

from omegaconf import DictConfig
from pytorch_lightning.loggers.logger import Logger


def get_logger(cfg: DictConfig) -> Logger:
    """Instantiate the logger specified by ``cfg.logger.type``.

    Reads ``cfg.logger.type`` and forwards the matching sub-config to the
    corresponding factory function.

    Args:
        cfg: Full Hydra config. Must contain a ``logger`` key with at least
            ``type`` and a matching sub-key (``tensorboard``, ``mlflow``,
            or ``wandb``) holding the logger hyperparameters.

    Returns:
        A configured PyTorch Lightning logger instance.

    Raises:
        ValueError: If ``cfg.logger.type`` is not one of the supported values.
    """
    logger_type = cfg.logger.type

    if logger_type == "tensorboard":
        from loggers.tensorboard_logger import build_logger

        params = cfg.logger.tensorboard
        return build_logger(
            save_dir=params.save_dir,
            name=params.name,
            version=params.version,
        )

    if logger_type == "mlflow":
        from loggers.mlflow_logger import build_logger

        params = cfg.logger.mlflow
        return build_logger(
            tracking_uri=params.tracking_uri,
            experiment_name=params.experiment_name,
            run_name=params.run_name,
        )

    if logger_type == "wandb":
        from loggers.wandb_logger import build_logger

        params = cfg.logger.wandb
        return build_logger(
            project=params.project,
            name=params.name,
            save_dir=params.save_dir,
        )

    raise ValueError(
        f"Unknown logger type: '{logger_type}'. "
        "Choose one of: tensorboard, mlflow, wandb."
    )
