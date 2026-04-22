"""Bridge utilities for reusing the Cityscapes seminar model."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_device(device_name: str) -> torch.device:
    """Resolve Hydra device string to a torch.device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _append_cityscapes_lightning_dir(cityscapes_project_dir: Path) -> Path:
    lightning_dir = (
        cityscapes_project_dir / "cityscapes-segmentation" / "lightning"
    ).resolve()
    if str(lightning_dir) not in sys.path:
        sys.path.insert(0, str(lightning_dir))
    return lightning_dir


def load_lightning_module(cityscapes_project_dir: str | Path):
    """Import the Cityscapes Lightning module after adding its local path."""
    _append_cityscapes_lightning_dir(Path(cityscapes_project_dir))
    return importlib.import_module("model")


def load_utils_module(cityscapes_project_dir: str | Path):
    """Import the Cityscapes utility module after adding its local path."""
    _append_cityscapes_lightning_dir(Path(cityscapes_project_dir))
    return importlib.import_module("utils")


def load_cityscapes_model(
    cityscapes_project_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the trained Cityscapes Lightning checkpoint."""
    model_module = load_lightning_module(cityscapes_project_dir)
    model = model_module.SegmentationModel.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model.eval()
    return model.to(device)


def build_preprocess_transform(
    cityscapes_project_dir: str | Path,
    height: int,
    width: int,
):
    """Build the validation-time Cityscapes preprocessing pipeline."""
    utils_module = load_utils_module(cityscapes_project_dir)
    return utils_module.build_transforms("val", height, width)


def collect_image_paths(input_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Collect image paths recursively in a deterministic order."""
    paths = [
        path
        for path in sorted(Path(input_dir).rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is None:
        return paths
    return paths[:limit]


def load_images_as_tensor_batch(
    image_paths: list[Path],
    transform,
) -> tuple[torch.Tensor, list[str]]:
    """Load images from disk and stack them into a BCHW tensor batch."""
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tensor = transform(image=image)["image"].float()
        tensors.append(tensor)
        names.append(image_path.name)
    if not tensors:
        raise ValueError("No images were loaded for benchmarking.")
    return torch.stack(tensors, dim=0), names


def make_random_input(
    batch_size: int,
    height: int,
    width: int,
    seed: int,
) -> torch.Tensor:
    """Create a reproducible synthetic tensor for export and smoke tests."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((batch_size, 3, height, width), generator=generator)
