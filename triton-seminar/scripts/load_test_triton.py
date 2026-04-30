#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient

RAW_FIELDS = [
    "timestamp",
    "model_name",
    "request_id",
    "concurrency",
    "batch_size",
    "latency_ms",
    "output_shape",
    "success",
    "error",
]

SUMMARY_FIELDS = [
    "timestamp",
    "model_name",
    "concurrency",
    "batch_size",
    "num_requests",
    "successful_requests",
    "failed_requests",
    "total_time_s",
    "throughput_rps",
    "avg_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load-test a Triton HTTP model from local clients."
    )
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model-name", default="cityscapes")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-csv", default="results/load_test_results.csv")
    parser.add_argument("--raw-csv", default="results/load_test_raw.csv")
    parser.add_argument("--summary-csv", default="results/load_test_summary.csv")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def run_request(
    request_id: int,
    args: argparse.Namespace,
    client_local: threading.local,
) -> dict[str, object]:
    if not hasattr(client_local, "client"):
        client_local.client = httpclient.InferenceServerClient(url=args.url)

    x = np.random.random((args.batch_size, 3, args.height, args.width)).astype(
        np.float32
    )
    infer_input = httpclient.InferInput("input", x.shape, "FP32")
    infer_input.set_data_from_numpy(x)
    requested_output = httpclient.InferRequestedOutput("logits")

    started = time.perf_counter()
    timestamp = utc_now()
    try:
        result = client_local.client.infer(
            model_name=args.model_name,
            inputs=[infer_input],
            outputs=[requested_output],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        output = result.as_numpy("logits")
        if output is None:
            raise RuntimeError("Triton response did not contain output 'logits'")
        return {
            "timestamp": timestamp,
            "model_name": args.model_name,
            "request_id": request_id,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "latency_ms": f"{latency_ms:.3f}",
            "output_shape": "x".join(str(dim) for dim in output.shape),
            "success": True,
            "error": "",
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "timestamp": timestamp,
            "model_name": args.model_name,
            "request_id": request_id,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "latency_ms": f"{latency_ms:.3f}",
            "output_shape": "",
            "success": False,
            "error": str(exc),
        }


def summarize(
    args: argparse.Namespace, rows: list[dict[str, object]], total_time_s: float
) -> dict[str, object]:
    latencies = [float(row["latency_ms"]) for row in rows if row["success"] is True]
    success_count = len(latencies)
    failed_count = len(rows) - success_count
    return {
        "timestamp": utc_now(),
        "model_name": args.model_name,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "num_requests": args.num_requests,
        "successful_requests": success_count,
        "failed_requests": failed_count,
        "total_time_s": f"{total_time_s:.3f}",
        "throughput_rps": f"{success_count / total_time_s:.3f}"
        if total_time_s > 0
        else "0.000",
        "avg_latency_ms": f"{statistics.mean(latencies):.3f}" if latencies else "0.000",
        "p50_latency_ms": f"{percentile(latencies, 0.50):.3f}",
        "p90_latency_ms": f"{percentile(latencies, 0.90):.3f}",
        "p95_latency_ms": f"{percentile(latencies, 0.95):.3f}",
        "p99_latency_ms": f"{percentile(latencies, 0.99):.3f}",
    }


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_requests < 1:
        raise ValueError("--num-requests must be at least 1")

    os.makedirs("results", exist_ok=True)
    client_local = threading.local()
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_request, request_id, args, client_local)
            for request_id in range(args.num_requests)
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    total_time_s = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["request_id"]))

    raw_path = Path(args.raw_csv)
    summary_path = Path(args.summary_csv)
    append_rows(raw_path, RAW_FIELDS, rows)
    summary = summarize(args, rows, total_time_s)
    append_rows(summary_path, SUMMARY_FIELDS, [summary])

    output_csv = Path(args.output_csv)
    if output_csv != raw_path:
        append_rows(output_csv, RAW_FIELDS, rows)

    print(
        "model={model_name} concurrency={concurrency} batch={batch_size} "
        "success={successful_requests}/{num_requests} throughput={throughput_rps} rps "
        "p95={p95_latency_ms} ms".format(**summary)
    )


if __name__ == "__main__":
    main()
