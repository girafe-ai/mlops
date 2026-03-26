"""MLflow logger factory."""

from pytorch_lightning.loggers import MLFlowLogger


def build_logger(
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
) -> MLFlowLogger:
    """Instantiate an MLFlowLogger from config parameters.

    Args:
        tracking_uri: MLflow tracking server URI
            (e.g. ``"http://localhost:5000"`` or a local path).
        experiment_name: Name of the MLflow experiment to log into.
        run_name: Display name for this run. ``None`` lets MLflow generate one.

    Returns:
        Configured :class:`MLFlowLogger`.
    """
    return MLFlowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
