from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.1"))
DEPTH = int(os.environ.get("DEPTH", "6"))
ITERATIONS = int(os.environ.get("ITERATIONS", "500"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))


def load_data():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test, data.feature_names


def train(X_train, y_train):
    model = CatBoostRegressor(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED,
        verbose=100,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return y_pred, metrics


def save_plots(y_test, y_pred, model, feature_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predictions vs Actual")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictions_vs_actual.png", dpi=150)
    plt.close(fig)

    importances = model.get_feature_importance()
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Hyperparameters: lr={LEARNING_RATE}, depth={DEPTH}, iterations={ITERATIONS}"
    )
    print(f"Output directory: {OUTPUT_DIR}")

    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = train(X_train, y_train)

    y_pred, metrics = evaluate(model, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")

    model.save_model(str(OUTPUT_DIR / "model.cbm"))
    print(f"Model saved to {OUTPUT_DIR / 'model.cbm'}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    save_plots(y_test, y_pred, model, feature_names)
    print(
        "Plots saved: predictions_vs_actual.png, feature_importance.png, residuals.png"
    )


if __name__ == "__main__":
    main()
