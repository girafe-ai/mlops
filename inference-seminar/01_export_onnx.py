"""Export the Cityscapes segmentation model to ONNX."""

from __future__ import annotations

import hydra
from inference_seminar.exporting import export_cityscapes_to_onnx
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    onnx_path, num_nodes, max_abs_diff = export_cityscapes_to_onnx(cfg)

    print(f"Saved ONNX model to: {onnx_path}")
    print(f"ONNX graph nodes: {num_nodes}")
    print(f"Max abs diff vs PyTorch: {max_abs_diff:.6e}")


if __name__ == "__main__":
    main()
