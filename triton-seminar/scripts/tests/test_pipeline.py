#!/usr/bin/env python3
import io
import urllib.request
from pathlib import Path

import hydra
import numpy as np
import tritonclient.http as httpclient
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def project_path(path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(get_original_cwd()) / candidate


def synthetic_jpeg_bytes() -> bytes:
    height, width = 512, 1024
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 128, dtype=np.uint8),
        ],
        axis=-1,
    )
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def read_image_bytes(path: str | None) -> tuple[bytes, str]:
    image_path = project_path(path)
    if image_path is None:
        return synthetic_jpeg_bytes(), "synthetic JPEG"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")
    return image_path.read_bytes(), str(image_path)


def print_pipeline_metrics(metrics_url: str) -> None:
    metrics = urllib.request.urlopen(metrics_url, timeout=5).read().decode("utf-8")
    model_names = (
        "cityscapes_pipeline",
        "image_preprocess",
        "cityscapes_embedder",
        "segmentation_postprocess",
    )
    for line in metrics.splitlines():
        if "nv_inference" in line and any(name in line for name in model_names):
            print(line)


@hydra.main(config_path="conf", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    encoded_bytes, image_source = read_image_bytes(cfg.image)
    encoded = np.frombuffer(encoded_bytes, dtype=np.uint8)

    client = httpclient.InferenceServerClient(url=cfg.url)
    infer_input = httpclient.InferInput("IMAGE_BYTES", encoded.shape, "UINT8")
    infer_input.set_data_from_numpy(encoded)
    outputs = [
        httpclient.InferRequestedOutput("PRED_MASK"),
        httpclient.InferRequestedOutput("COLOR_MASK"),
        httpclient.InferRequestedOutput("CLASS_HISTOGRAM"),
        httpclient.InferRequestedOutput("MEAN_CONFIDENCE"),
    ]

    result = client.infer(cfg.model_name, inputs=[infer_input], outputs=outputs)
    pred_mask = result.as_numpy("PRED_MASK")
    color_mask = result.as_numpy("COLOR_MASK")
    histogram = result.as_numpy("CLASS_HISTOGRAM")
    mean_confidence = result.as_numpy("MEAN_CONFIDENCE")

    print(f"model name: {cfg.model_name}")
    print(f"image source: {image_source}")
    print(f"image bytes: {encoded.shape[0]}")
    print(f"pred mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
    print(f"color mask shape: {color_mask.shape}, dtype: {color_mask.dtype}")
    print(f"class histogram: {histogram.tolist()}")
    print(f"mean confidence: {float(mean_confidence[0]):.6f}")

    if cfg.save_color_mask:
        output_path = project_path(cfg.save_color_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color_mask[0], mode="RGB").save(output_path)
        print(f"saved color mask: {output_path}")

    if cfg.print_metrics:
        print_pipeline_metrics(cfg.metrics_url)


if __name__ == "__main__":
    main()
