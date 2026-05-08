"""Send a local image to an MLflow model serving endpoint."""

from __future__ import annotations

import base64
from pathlib import Path

import hydra
import requests
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    image_path = resolve_path(cfg.test.image)
    if cfg.test.input_mode == "path":
        record = {"image_path": str(image_path)}
    elif cfg.test.input_mode == "base64":
        record = {
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii")
        }
    else:
        raise ValueError("test.input_mode must be either 'base64' or 'path'")

    response = requests.post(
        cfg.test.url,
        json={"dataframe_records": [record]},
        timeout=120,
    )
    response.raise_for_status()
    prediction = response.json()["predictions"][0]
    print("mean_confidence:", prediction["mean_confidence"])
    print("image_size:", prediction["width"], "x", prediction["height"])
    print("classes:", ", ".join(prediction["class_histogram"].keys()))


if __name__ == "__main__":
    main()
