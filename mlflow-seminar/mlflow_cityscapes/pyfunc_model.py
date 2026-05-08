"""MLflow PyFunc wrapper for an existing Cityscapes segmentation checkpoint."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_TRAIN_ID_TO_COLOR = np.asarray(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_base64(value: str) -> Image.Image:
    if "," in value:
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class CityscapesSegmentationPyFunc(mlflow.pyfunc.PythonModel):
    """Serve a pre-trained Lightning Cityscapes model through MLflow.

    The model accepts a pandas DataFrame with either ``image_base64`` or
    ``image_path`` and returns display-ready base64 PNGs plus compact
    segmentation metadata.
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 1024,
        device: str = "auto",
    ) -> None:
        self.height = height
        self.width = width
        self.device_name = device

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        lightning_code = Path(context.artifacts["lightning_code"])
        sys.path.insert(0, str(lightning_code))

        from model import SegmentationModel
        from utils import build_transforms

        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)

        self.model = SegmentationModel.load_from_checkpoint(
            context.artifacts["checkpoint"],
            map_location=self.device,
            encoder_weights=None,
        )
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms("val", self.height, self.width)

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        del context
        rows = self._normalize_input(model_input)
        return pd.DataFrame([self._predict_one(row) for row in rows])

    def _normalize_input(
        self,
        model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    ) -> list[dict[str, Any]]:
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        if isinstance(model_input, dict):
            if "image_base64" in model_input or "image_path" in model_input:
                return [model_input]
            if "inputs" in model_input:
                return model_input["inputs"]
        return list(model_input)

    def _predict_one(self, row: dict[str, Any]) -> dict[str, Any]:
        image = self._load_image(row)
        original = image.resize((self.width, self.height), Image.BILINEAR)
        image_np = np.asarray(original, dtype=np.uint8)
        tensor = self.transform(image=image_np)["image"].unsqueeze(0).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            probs = torch.softmax(logits, dim=1).max(dim=1).values
            mean_confidence = float(probs.mean().cpu().item())

        color_mask_np = CITYSCAPES_TRAIN_ID_TO_COLOR[pred]
        color_mask = Image.fromarray(color_mask_np, mode="RGB")
        overlay = Image.blend(original, color_mask, alpha=0.45)
        counts = np.bincount(pred.reshape(-1), minlength=len(CITYSCAPES_CLASSES))
        total = int(counts.sum())

        class_histogram = {
            name: {
                "pixels": int(count),
                "share": round(float(count) / total, 5) if total else 0.0,
            }
            for name, count in zip(CITYSCAPES_CLASSES, counts)
            if count > 0
        }

        return {
            "mask_base64": encode_png(color_mask),
            "overlay_base64": encode_png(overlay),
            "width": self.width,
            "height": self.height,
            "mean_confidence": round(mean_confidence, 5),
            "class_histogram": class_histogram,
        }

    def _load_image(self, row: dict[str, Any]) -> Image.Image:
        image_base64 = row.get("image_base64")
        if isinstance(image_base64, str) and image_base64:
            return decode_image_base64(image_base64)

        image_path = row.get("image_path")
        if isinstance(image_path, str) and image_path:
            return Image.open(image_path).convert("RGB")

        raise ValueError(
            "Expected either image_base64 or image_path in each request row."
        )
