#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_CSV = Path("results/load_test_summary.csv")
PLOTS_DIR = Path("results/plots")
TABLE_MD = Path("results/summary_table.md")


PLOTS = [
    ("throughput_rps", "Throughput, requests/s", "throughput_vs_concurrency.png"),
    ("avg_latency_ms", "Average latency, ms", "avg_latency_vs_concurrency.png"),
    ("p95_latency_ms", "P95 latency, ms", "p95_latency_vs_concurrency.png"),
    ("p99_latency_ms", "P99 latency, ms", "p99_latency_vs_concurrency.png"),
]


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"{SUMMARY_CSV} does not exist; run load tests first")

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        raise ValueError(f"{SUMMARY_CSV} has no rows; run load tests first")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (model_name, batch_size), group in df.groupby(["model_name", "batch_size"]):
            group = group.sort_values("concurrency")
            ax.plot(
                group["concurrency"],
                group[metric],
                marker="o",
                label=f"{model_name}, batch={batch_size}",
            )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    table_columns = [
        "model_name",
        "concurrency",
        "batch_size",
        "successful_requests",
        "failed_requests",
        "throughput_rps",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
    ]
    TABLE_MD.write_text(df[table_columns].to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
