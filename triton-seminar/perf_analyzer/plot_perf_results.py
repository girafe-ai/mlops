#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("perf_analyzer/results")
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARY_MD = RESULTS_DIR / "perf_summary.md"


def find_column(
    df: pd.DataFrame, candidates: tuple[str, ...], contains: tuple[str, ...] = ()
) -> str:
    normalized = {column.strip().lower(): column for column in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    if contains:
        for column in df.columns:
            lowered = column.strip().lower()
            if all(part in lowered for part in contains):
                return column

    raise KeyError(f"Could not find column. Available columns: {list(df.columns)}")


def load_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("*_perf.csv")):
        model_name = path.stem.removesuffix("_perf")
        df = pd.read_csv(path)
        if df.empty:
            continue

        concurrency_col = find_column(df, ("Concurrency",))
        throughput_col = find_column(
            df,
            ("Inferences/Second", "Inferences / Second"),
            contains=("infer", "second"),
        )
        latency_col = find_column(
            df,
            ("p95 latency"),
            contains=("p95", "latency"),
        )

        slim = pd.DataFrame(
            {
                "model_name": model_name,
                "concurrency": pd.to_numeric(df[concurrency_col]),
                "throughput": pd.to_numeric(df[throughput_col]),
                "latency_p95_us": pd.to_numeric(df[latency_col]),
            }
        )
        slim["latency_p95_ms"] = slim["latency_p95_us"] / 1000.0
        frames.append(slim)

    if not frames:
        raise FileNotFoundError(f"No Perf Analyzer CSV files found in {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in df.groupby("model_name"):
        group = group.sort_values("concurrency")
        print(group)
        ax.plot(group["concurrency"], group[metric], marker="o", label=model_name)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_results()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric(
        df,
        metric="latency_p95_ms",
        ylabel="p95 latency, ms",
        filename="latency_avg_vs_concurrency.png",
    )
    plot_metric(
        df,
        metric="throughput",
        ylabel="Throughput, inferences/s",
        filename="throughput_vs_concurrency.png",
    )

    table = df.sort_values(["model_name", "concurrency"])
    SUMMARY_MD.write_text(table.to_markdown(index=False) + "\n")
    print(f"Wrote {PLOTS_DIR / 'latency_avg_vs_concurrency.png'}")
    print(f"Wrote {PLOTS_DIR / 'throughput_vs_concurrency.png'}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
