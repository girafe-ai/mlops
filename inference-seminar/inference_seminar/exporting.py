"""Reusable ONNX export utilities for the seminar."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig

from inference_seminar.cityscapes_adapter import (
    build_preprocess_transform,
    collect_image_paths,
    load_cityscapes_model,
    load_images_as_tensor_batch,
    make_random_input,
    resolve_device,
    resolve_repo_path,
)


def export_cityscapes_to_onnx(cfg: DictConfig) -> tuple[Path, int, float]:
    """Export the configured Cityscapes checkpoint to ONNX and validate parity."""
    artifacts_dir = resolve_repo_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.model.device)
    model = load_cityscapes_model(
        cfg.paths.cityscapes_project_dir,
        cfg.paths.cityscapes_checkpoint,
        device,
    )

    if cfg.export.input_mode == "images":
        transform = build_preprocess_transform(
            cfg.paths.cityscapes_project_dir,
            cfg.model.height,
            cfg.model.width,
        )
        image_paths = collect_image_paths(
            cfg.paths.cityscapes_input_dir,
            limit=cfg.model.batch_size,
        )
        batch, _ = load_images_as_tensor_batch(image_paths, transform)
    else:
        batch = make_random_input(
            batch_size=cfg.model.batch_size,
            height=cfg.model.height,
            width=cfg.model.width,
            seed=cfg.export.seed,
        )

    batch = batch.to(device)
    onnx_path = resolve_repo_path(cfg.export.onnx_path)
    dynamic_axes = None
    if cfg.model.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            batch,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=int(cfg.model.opset),
            do_constant_folding=True,
        )
        torch_output = model(batch).detach().cpu().numpy()

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_output = ort_session.run(None, {"input": batch.detach().cpu().numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))

    torch.testing.assert_close(
        torch.from_numpy(ort_output),
        torch.from_numpy(torch_output),
        atol=float(cfg.export.parity_atol),
        rtol=float(cfg.export.parity_rtol),
    )
    return onnx_path, len(onnx_model.graph.node), max_abs_diff
