# Perf Analyzer

`perf_analyzer` is Triton's native benchmarking tool. It sends repeated requests to a served model and reports throughput, average latency, and latency percentiles for each concurrency level.

## Start Triton

Run the single-model repository first:

```bash
sudo docker compose up triton
```

This serves:

- `cityscapes`: baseline model without Triton dynamic batching.
- `cityscapes_dyn`: dynamic-batching model.

## Run Benchmarks

From the repository root:

```bash
bash perf_analyzer/run_perf_analyzer.sh
```

The runner benchmarks both models over concurrency `1, 2, 4, 8` and writes CSV files to:

```text
perf_analyzer/results/
```

You can override defaults:

```bash
CONCURRENCY_RANGE=1:16:1 MEASUREMENT_INTERVAL_MS=10000 bash perf_analyzer/run_perf_analyzer.sh
```

If your Docker setup does not need `sudo`, use:

```bash
DOCKER_COMPOSE="docker compose" bash perf_analyzer/run_perf_analyzer.sh
```

## Plot

```bash
python perf_analyzer/plot_perf_results.py
```

Outputs:

- `perf_analyzer/results/plots/latency_avg_vs_concurrency.png`
- `perf_analyzer/results/plots/throughput_vs_concurrency.png`
- `perf_analyzer/results/perf_summary.md`

## Notes

The baseline model has `max_batch_size: 0`, so the request shape includes the leading dimension:

```text
input:1,3,1024,2048
```

The dynamic model has `max_batch_size > 0`, so Triton owns the batch dimension. Perf Analyzer uses `--batch-size 1` and the tensor shape excludes batch:

```text
input:3,1024,2048
```

Useful CSV columns:

- `Concurrency`
- `Inferences/Second`
- average latency column, depending on Triton version
- `p95 latency`
- `p99 latency`

Use the Prometheus metrics endpoint after benchmarks to inspect whether dynamic batching reduced backend execution count:

```bash
curl localhost:8002/metrics | grep -E 'cityscapes|nv_inference_exec_count|nv_inference_request_success'
```
