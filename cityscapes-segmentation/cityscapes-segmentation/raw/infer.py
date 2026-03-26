"""Run inference with a trained Cityscapes segmentation model.

Usage:

    uv run python infer.py
"""

import logging
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from cityscapesscripts.helpers.labels import labels as cs_labels
from data import NUM_CLASSES
from PIL import Image
from tqdm import tqdm
from utils import build_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CHECKPOINT = "checkpoints/best.pth"
INPUT = "../../data/leftImg8bit/val"
OUTPUT_DIR = "inference_results"
ENCODER = "resnet34"
HEIGHT = 512
WIDTH = 1024
DEVICE = "auto"
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# trainId → RGB color from Cityscapes label definitions
TRAIN_ID_TO_COLOR = {}
for _label in cs_labels:
    if _label.trainId not in (-1, 255) and _label.trainId not in TRAIN_ID_TO_COLOR:
        TRAIN_ID_TO_COLOR[_label.trainId] = _label.color

# Unlabeled / void pixels rendered as black
TRAIN_ID_TO_COLOR[255] = (0, 0, 0)

_COLOR_PALETTE = np.zeros((256, 3), dtype=np.uint8)
for _tid, _color in TRAIN_ID_TO_COLOR.items():
    _COLOR_PALETTE[_tid] = _color


def _load_model(checkpoint: str, encoder: str, device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name=encoder, encoder_weights=None, in_channels=3, classes=NUM_CLASSES
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def _predict_image(
    model: torch.nn.Module,
    img_path: Path,
    transform: A.Compose,
    device: torch.device,
) -> np.ndarray:
    """Return an H×W uint8 array of predicted train IDs."""
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    tensor = transform(image=image)["image"].unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _colorize(pred: np.ndarray) -> Image.Image:
    """Map predicted train IDs to an RGB image using Cityscapes colors."""
    h, w = pred.shape
    rgb = _COLOR_PALETTE[pred.ravel()].reshape(h, w, 3)
    return Image.fromarray(rgb)


def run(
    checkpoint: str = CHECKPOINT,
    input: str = INPUT,
    output_dir: str | None = OUTPUT_DIR,
    encoder: str = ENCODER,
    height: int = HEIGHT,
    width: int = WIDTH,
    device: str = DEVICE,
) -> None:
    """Run inference on a single image or a directory of images.

    Args:
        checkpoint: Path to the checkpoint produced by ``train.py``.
        input: Path to an image file or a directory containing images.
        output_dir: Directory to write ``*_pred.png`` results.
            If ``None``, results are saved alongside the source images.
        encoder: Encoder backbone matching the one used during training.
        height: Resize height applied before inference.
        width: Resize width applied before inference.
        device: ``"auto"`` selects CUDA when available, or ``"cpu"``/``"cuda"``.
    """
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    log.info("Using device: %s", _device)
    log.info("Loading checkpoint: %s", checkpoint)
    model = _load_model(checkpoint, encoder, _device)
    transform = build_transforms("val", height, width)

    input_path = Path(input)
    if input_path.is_file():
        img_paths = [input_path]
    elif input_path.is_dir():
        img_paths = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.suffix.lower() in _IMG_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not img_paths:
        log.warning("No images found at %s", input_path)
        return

    out_root = Path(output_dir) if output_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    log.info("Running inference on %d image(s)", len(img_paths))
    for img_path in tqdm(img_paths, desc="Inference", unit="img"):
        pred = _predict_image(model, img_path, transform, _device)
        color_map = _colorize(pred)

        if out_root:
            out_path = out_root / (img_path.stem + "_pred.png")
        else:
            out_path = img_path.with_name(img_path.stem + "_pred.png")

        color_map.save(out_path)
        log.debug("%s  -->  %s", img_path.name, out_path)

    log.info("Done. %d image(s) processed.", len(img_paths))


if __name__ == "__main__":
    run()
