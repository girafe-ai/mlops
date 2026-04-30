"""Train a boosted-tree model and compare XGBoost inference with RAPIDS FIL."""

import time
from pathlib import Path
from typing import Callable

import cupy as cp
import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.fil import ForestInference, set_fil_device_type
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEMINAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEMINAR_ROOT.parent


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve seminar config paths against the repository root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def measure(fn: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Return mean latency in ms and throughput in batches/s."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - start) * 1_000.0)

    mean_ms = float(np.mean(durations))
    if mean_ms <= 0:
        raise RuntimeError(f"Measured non-positive latency: {mean_ms}")

    batches_per_s = 1_000.0 / mean_ms
    return mean_ms, batches_per_s


def cuda_available() -> bool:
    """Return True if at least one CUDA device is visible to CuPy."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("Failed to query CUDA devices with CuPy.") from exc


def resolve_xgb_device(configured_device: str) -> str:
    """Resolve the configured XGBoost device."""
    if configured_device == "auto":
        return "cuda" if cuda_available() else "cpu"

    if configured_device not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device {configured_device!r}. "
            "Expected 'auto', 'cpu', or 'cuda'."
        )

    if configured_device == "cuda" and not cuda_available():
        raise RuntimeError(
            "Configured device is 'cuda', but no CUDA device is available."
        )

    return configured_device


def prepare_xgb_batch(batch: np.ndarray, device: str) -> np.ndarray | cp.ndarray:
    """Convert the batch to a device-appropriate container for XGBoost."""
    if device == "cuda":
        return cp.asarray(batch)
    return batch


def load_fil_model(model_path: Path, device: str) -> ForestInference:
    """Load a FIL model on the requested device."""
    if device == "cpu":
        with set_fil_device_type("cpu"):
            return ForestInference.load(
                str(model_path),
                output_type="numpy",
                is_classifier=True,
            )

    if device == "cuda":
        return ForestInference.load(
            str(model_path),
            output_type="numpy",
            is_classifier=True,
        )

    raise ValueError(f"Unsupported FIL device {device!r}. Expected 'cpu' or 'cuda'.")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = resolve_repo_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_repo_path(cfg.fil.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = resolve_repo_path(cfg.fil.results_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=int(cfg.fil.dataset.n_samples),
        n_features=int(cfg.fil.dataset.n_features),
        n_informative=int(cfg.fil.dataset.n_informative),
        n_redundant=0,
        n_classes=int(cfg.fil.dataset.n_classes),
        random_state=int(cfg.fil.dataset.random_state),
    )
    X = X.astype(np.float32)

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=int(cfg.fil.dataset.random_state),
        stratify=y,
    )

    xgb_device = resolve_xgb_device(str(cfg.fil.device))

    model = xgb.XGBClassifier(
        max_depth=int(cfg.fil.training.max_depth),
        n_estimators=int(cfg.fil.training.n_estimators),
        learning_rate=float(cfg.fil.training.learning_rate),
        subsample=float(cfg.fil.training.subsample),
        colsample_bytree=float(cfg.fil.training.colsample_bytree),
        objective="multi:softprob",
        num_class=int(cfg.fil.dataset.n_classes),
        tree_method="hist",
        device=xgb_device,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    model.save_model(model_path)

    batch_size = int(cfg.fil.inference.batch_size)
    repeats = int(cfg.fil.inference.repeats)

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch = X_test[:batch_size]
    if len(batch) == 0:
        raise RuntimeError("Inference batch is empty.")

    xgb_batch = prepare_xgb_batch(batch, xgb_device)
    xgb_mean_ms, xgb_batches_per_s = measure(
        lambda: model.predict_proba(xgb_batch), repeats
    )

    rows = [
        {
            "backend": "xgboost",
            "available": True,
            "mean_ms": xgb_mean_ms,
            "throughput_batches_per_s": xgb_batches_per_s,
            "throughput_items_per_s": xgb_batches_per_s * len(batch),
            "notes": f"device={xgb_device}",
        }
    ]

    fil_device = xgb_device
    fil_model = load_fil_model(model_path, fil_device)
    fil_model.optimize(batch_size=batch_size)

    # Check which hyperparameters were selected
    print(f"Layout: {fil_model.layout}")
    print(f"Chunk size: {fil_model.default_chunk_size}")

    fil_mean_ms, fil_batches_per_s = measure(
        lambda: fil_model.predict_proba(batch), repeats
    )

    rows.append(
        {
            "backend": "fil",
            "available": True,
            "mean_ms": fil_mean_ms,
            "throughput_batches_per_s": fil_batches_per_s,
            "throughput_items_per_s": fil_batches_per_s * len(batch),
            "notes": f"device={fil_device}",
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved FIL benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
