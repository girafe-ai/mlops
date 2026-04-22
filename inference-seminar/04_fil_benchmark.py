"""Train a boosted-tree model and compare XGBoost inference with RAPIDS FIL."""

import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from inference_seminar.cityscapes_adapter import resolve_repo_path
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def measure(fn, repeats: int) -> tuple[float, float]:
    """Return mean latency in ms and throughput in items/s."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - start) * 1_000.0)
    mean_ms = float(np.mean(durations))
    return mean_ms, 1_000.0 / mean_ms


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = resolve_repo_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_repo_path(cfg.fil.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

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
    xgb_device = cfg.fil.device
    if xgb_device == "auto":
        xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

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
    batch = X_test[:batch_size]

    xgb_mean_ms, xgb_items_per_s = measure(lambda: model.predict_proba(batch), repeats)
    rows = [
        {
            "backend": "xgboost",
            "available": True,
            "mean_ms": xgb_mean_ms,
            "throughput_batches_per_s": xgb_items_per_s,
            "throughput_items_per_s": xgb_items_per_s * len(batch),
            "notes": f"device={xgb_device}",
        }
    ]

    try:
        from cuml.fil import ForestInference

        fil_device = xgb_device if xgb_device in {"cpu", "cuda"} else "cuda"
        if fil_device == "cpu":
            from cuml.fil import set_fil_device_type

            with set_fil_device_type("cpu"):
                fil_model = ForestInference.load(str(model_path), output_type="numpy")
        else:
            fil_model = ForestInference.load(str(model_path), output_type="numpy")
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
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "backend": "fil",
                "available": False,
                "mean_ms": None,
                "throughput_batches_per_s": None,
                "throughput_items_per_s": None,
                "notes": str(exc),
            }
        )

    df = pd.DataFrame(rows)
    output_path = resolve_repo_path(cfg.fil.results_csv)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved FIL benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
