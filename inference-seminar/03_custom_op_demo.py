"""Demonstrate a custom ONNX operator executed via ONNX Runtime Extensions."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import DictConfig
from onnx import TensorProto, helper
from onnxruntime_extensions import PyCustomOpDef, get_library_path, onnx_op


@onnx_op(
    op_type="AddOne",
    inputs=[PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def add_one(x):
    """Small custom op used in the seminar to prove the extension path works."""
    return x + 1.0


def build_model(model_path: Path) -> None:
    """Create a tiny ONNX graph that calls the custom operator."""
    input_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N"])
    output_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N"])
    node = helper.make_node("AddOne", ["x"], ["y"], domain="ai.onnx.contrib")
    graph = helper.make_graph(
        [node], "custom_add_one_graph", [input_tensor], [output_tensor]
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("ai.onnx.contrib", 1),
        ],
        producer_name="inference-seminar",
    )
    onnx.save(model, str(model_path))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model_path = Path(cfg.custom_op.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    build_model(model_path)

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(get_library_path())
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_values = np.asarray(cfg.custom_op.input_values, dtype=np.float32)
    output_values = session.run(None, {"x": input_values})[0]
    deltas = output_values - input_values

    if not np.allclose(deltas, float(cfg.custom_op.expected_delta)):
        raise AssertionError(
            "Custom operator output does not match the expected delta."
        )

    print(f"Saved custom-op model to: {model_path}")
    print(f"Input : {input_values.tolist()}")
    print(f"Output: {output_values.tolist()}")


if __name__ == "__main__":
    main()
