"""Small helpers for inference timing and statistics."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class TimingStats:
    """Summary statistics for repeated inference measurements."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_items_per_s: float


def synchronize_if_needed(device: torch.device | None) -> None:
    """Synchronize CUDA work before reading host-side timers."""
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def measure_callable(
    fn: Callable[[], object],
    *,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    device: torch.device | None = None,
) -> TimingStats:
    """Benchmark a zero-argument inference callable."""
    for _ in range(warmup_runs):
        fn()
        synchronize_if_needed(device)

    durations_ms: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        fn()
        synchronize_if_needed(device)
        durations_ms.append((time.perf_counter() - start) * 1_000.0)

    mean_ms = statistics.fmean(durations_ms)
    median_ms = statistics.median(durations_ms)
    p95_ms = float(np.percentile(durations_ms, 95))
    throughput = batch_size / (mean_ms / 1_000.0)
    return TimingStats(
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        throughput_items_per_s=throughput,
    )
