"""Register an existing Cityscapes segmentation checkpoint in MLflow."""

from __future__ import annotations

import base64
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow_cityscapes.pyfunc_model import CityscapesSegmentationPyFunc
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def read_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def find_registered_version(client: MlflowClient, model_name: str, run_id: str) -> str:
    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return version.version
    raise RuntimeError(f"Could not find registered version for run {run_id}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    checkpoint = resolve_path(cfg.register.checkpoint)
    lightning_code = resolve_path(cfg.register.lightning_code)
    example_image = resolve_path(cfg.register.example_image)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Override register.checkpoint=path/to/best.ckpt or place it at the default path."
        )
    if not lightning_code.exists():
        raise FileNotFoundError(f"Lightning code directory not found: {lightning_code}")
    if not example_image.exists():
        raise FileNotFoundError(f"Example image not found: {example_image}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    example = pd.DataFrame([{"image_base64": read_base64(example_image)}])
    signature = infer_signature(example)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_params(
            {
                "model_task": "semantic_segmentation",
                "dataset": "cityscapes",
                "height": cfg.model.height,
                "width": cfg.model.width,
                "checkpoint": str(checkpoint),
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CityscapesSegmentationPyFunc(
                height=cfg.model.height,
                width=cfg.model.width,
                device=cfg.model.device,
            ),
            artifacts={
                "checkpoint": str(checkpoint),
                "lightning_code": str(lightning_code),
            },
            input_example=example,
            signature=signature,
            registered_model_name=cfg.mlflow.model_name,
        )

    client = MlflowClient()
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        version = find_registered_version(
            client, cfg.mlflow.model_name, run.info.run_id
        )
    client.set_registered_model_alias(
        cfg.mlflow.model_name,
        cfg.mlflow.alias,
        str(version),
    )

    print(f"Registered model: {cfg.mlflow.model_name}")
    print(f"Version: {version}")
    print(f"Alias: {cfg.mlflow.alias}")
    print(f"Serve URI: models:/{cfg.mlflow.model_name}@{cfg.mlflow.alias}")


if __name__ == "__main__":
    main()
