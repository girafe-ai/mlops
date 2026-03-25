"""Run inference with a trained Cityscapes segmentation model.

Usage:

    uv run python infer.py
    uv run python infer.py inference.checkpoint=checkpoints/best.ckpt
    uv run python infer.py inference.input=../data/leftImg8bit/val inference.output_dir=predictions
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from cityscapesscripts.helpers.labels import labels as cs_labels
from model import SegmentationModel
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from utils import build_transforms

log = logging.getLogger(__name__)

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# trainId → RGB color from Cityscapes label definitions
TRAIN_ID_TO_COLOR: dict[int, tuple[int, int, int]] = {}
for _label in cs_labels:
    if _label.trainId not in (-1, 255) and _label.trainId not in TRAIN_ID_TO_COLOR:
        TRAIN_ID_TO_COLOR[_label.trainId] = _label.color
TRAIN_ID_TO_COLOR[255] = (0, 0, 0)

_COLOR_PALETTE = np.zeros((256, 3), dtype=np.uint8)
for _tid, _color in TRAIN_ID_TO_COLOR.items():
    _COLOR_PALETTE[_tid] = _color


def _colorize(pred: np.ndarray) -> Image.Image:
    h, w = pred.shape
    rgb = _COLOR_PALETTE[pred.ravel()].reshape(h, w, 3)
    return Image.fromarray(rgb)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    """Run inference on a single image or directory of images.

    Args:
        cfg: Hydra config composed from conf/config.yaml and its defaults.
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.inference.device == "auto"
        else torch.device(cfg.inference.device)
    )
    log.info("Using device: %s", device)
    log.info("Loading checkpoint: %s", cfg.inference.checkpoint)

    model = SegmentationModel.load_from_checkpoint(
        cfg.inference.checkpoint, map_location=device
    )
    model.eval()

    transform = build_transforms("val", cfg.data.height, cfg.data.width)

    input_path = Path(cfg.inference.input)
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

    out_root = Path(cfg.inference.output_dir) if cfg.inference.output_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    log.info("Running inference on %d image(s)", len(img_paths))
    for img_path in tqdm(img_paths, desc="Inference", unit="img"):
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        tensor = transform(image=image)["image"].unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        color_map = _colorize(pred)

        out_path = (
            out_root / (img_path.stem + "_pred.png")
            if out_root
            else img_path.with_name(img_path.stem + "_pred.png")
        )
        color_map.save(out_path)

    log.info("Done. %d image(s) processed.", len(img_paths))


if __name__ == "__main__":
    run()
