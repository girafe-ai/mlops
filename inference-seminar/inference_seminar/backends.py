"""Inference backends used by the seminar benchmarks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


@dataclass
class BackendResult:
    """Materialized benchmark outcome for a single backend."""

    backend: str
    available: bool
    provider: str
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    throughput_items_per_s: float | None
    max_abs_diff_vs_pytorch: float | None
    notes: str = ""


def get_available_ort_providers() -> list[str]:
    """Return available ONNX Runtime providers across minor API differences."""
    if hasattr(ort, "get_available_providers"):
        return list(ort.get_available_providers())

    if hasattr(ort, "get_all_providers"):
        return list(ort.get_all_providers())

    capi = getattr(ort, "capi", None)
    pybind_state = getattr(capi, "_pybind_state", None) if capi is not None else None
    if pybind_state is not None and hasattr(pybind_state, "get_available_providers"):
        return list(pybind_state.get_available_providers())

    raise AttributeError(
        "Could not discover ONNX Runtime providers. "
        "Please check `import onnxruntime as ort; print(ort.__file__, getattr(ort, '__version__', 'unknown'))` "
        "to confirm the expected package is installed."
    )


def ort_session_for_backend(
    backend: str,
    model_path: str | Path,
    *,
    trt_engine_cache_path: str | Path | None = None,
) -> tuple[ort.InferenceSession | None, str, str]:
    """Build an ONNX Runtime session for a named backend."""
    available = get_available_ort_providers()
    session_options = ort.SessionOptions()

    if backend == "cpu":
        provider_name = "CPUExecutionProvider"
        if provider_name not in available:
            return None, provider_name, "CPUExecutionProvider is not available."
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[provider_name],
        )
        return session, provider_name, ""

    if backend == "cuda":
        provider_name = "CUDAExecutionProvider"
        if provider_name not in available:
            return None, provider_name, "CUDAExecutionProvider is not available."
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[provider_name, "CPUExecutionProvider"],
        )
        return session, provider_name, ""

    if backend == "tensorrt":
        provider_name = "TensorrtExecutionProvider"
        if provider_name not in available:
            return None, provider_name, "TensorrtExecutionProvider is not available."
        provider_options = {
            "trt_engine_cache_enable": True,
            "trt_fp16_enable": True,
        }
        if trt_engine_cache_path is not None:
            provider_options["trt_engine_cache_path"] = str(trt_engine_cache_path)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[
                (provider_name, provider_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        return session, provider_name, ""

    raise ValueError(f"Unsupported backend: {backend}")


def run_ort_session(session: ort.InferenceSession, inputs: np.ndarray) -> np.ndarray:
    """Run an ORT session on a NumPy batch."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: inputs})[0]


def run_pytorch_model(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    """Run the PyTorch reference model and materialize a NumPy output."""
    with torch.inference_mode():
        return model(batch).detach().cpu().numpy()
