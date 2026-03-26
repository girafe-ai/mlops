"""TensorBoard logger factory."""

from pytorch_lightning.loggers import TensorBoardLogger


def build_logger(
    save_dir: str,
    name: str,
    version: str | int | None,
) -> TensorBoardLogger:
    """Instantiate a TensorBoardLogger from config parameters.

    Args:
        save_dir: Root directory where TensorBoard logs are saved.
        name: Experiment name (subdirectory under ``save_dir``).
        version: Run version string or integer. ``None`` auto-increments.

    Returns:
        Configured :class:`TensorBoardLogger`.
    """
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
    )
