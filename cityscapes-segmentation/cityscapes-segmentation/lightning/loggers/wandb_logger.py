"""Weights & Biases logger factory."""

from pytorch_lightning.loggers import WandbLogger


def build_logger(
    project: str,
    name: str | None,
    save_dir: str,
) -> WandbLogger:
    """Instantiate a WandbLogger from config parameters.

    Args:
        project: W&B project name to log into.
        name: Display name for this run. ``None`` lets W&B generate one.
        save_dir: Local directory where W&B writes run files.

    Returns:
        Configured :class:`WandbLogger`.
    """
    return WandbLogger(
        project=project,
        name=name,
        save_dir=save_dir,
    )
