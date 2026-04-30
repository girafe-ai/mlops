# Conclusions

## What Was Tested

This seminar setup compares two Triton models that use the same TensorRT plan:

- `cityscapes`: baseline model without Triton dynamic batching.
- `cityscapes_dyn`: dynamic-batching model with `max_batch_size: 2`, `preferred_batch_size: [ 2 ]`, and `max_queue_delay_microseconds: 10000`.

The intended workflow is:

1. Build a TensorRT engine from `assets/cityscapes_unet.onnx`.
2. Serve the TensorRT plan with Triton Inference Server.
3. Generate local concurrent HTTP inference load.
4. Collect Triton Prometheus metrics.
5. Plot throughput and latency versus concurrency.

## Key Observations

No runtime load-test values have been collected yet in this repository snapshot. After running experiments, use actual values from `results/load_test_summary.csv` to compare:

- Throughput as concurrency increases.
- Average latency as concurrency increases.
- P95 and P99 tail latency at higher concurrency.
- `cityscapes` versus `cityscapes_dyn` at the same concurrency and batch size.
- Queue duration and compute duration in Triton metrics.

## Dynamic Batching Check

Dynamic batching appears to be working when `cityscapes_dyn` has fewer backend executions than successful client requests under concurrent load. For example, about 20 successful requests with about 10 backend executions indicates Triton combined requests into batches of 2.

The most relevant metrics are:

- `nv_inference_request_success`
- `nv_inference_request_failure`
- `nv_inference_count`
- `nv_inference_exec_count`
- `nv_inference_request_duration_us`
- `nv_inference_queue_duration_us`
- `nv_inference_compute_infer_duration_us`

## Latency-Throughput Tradeoff

Low concurrency usually gives lower latency because requests spend less time waiting in Triton queues. Higher concurrency can improve throughput by keeping the GPU busier and by giving `cityscapes_dyn` enough simultaneous work to form dynamic batches. Too much concurrency can increase queueing and tail latency, so P95 and P99 latency matter more than average latency alone.

## Recommendations

- Start with `max_batch_size: 2` for 8 GB GPUs.
- Tune `max_queue_delay_microseconds` carefully.
- Use a low queue delay for online latency-sensitive inference.
- Use a higher queue delay for throughput-oriented workloads.
- Watch P95 and P99 latency, not only average latency.
- Monitor execution count versus request count to verify batching.

## Troubleshooting Notes

- If `assets/cityscapes_unet.onnx` is missing, place the ONNX model there before running `trtexec-build`.
- If `assets/cityscapes_unet.plan` is missing, run `sudo docker compose run --rm trtexec-build`.
- If batch size 2 fails, the GPU may not have enough VRAM. Rebuild with smaller profiles or benchmark batch size 1 only.
- If Triton reports a shape mismatch, check whether the target model is `cityscapes` or `cityscapes_dyn`. The baseline model has `max_batch_size: 0` and includes the batch-like dimension in `dims`; the dynamic model has `max_batch_size: 2` and excludes the batch dimension from `dims`.
- If Python clients cannot import Triton packages, run `pip install -r requirements-triton-client.txt`.
