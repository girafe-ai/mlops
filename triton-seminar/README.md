# Triton + TensorRT Cityscapes Seminar

This seminar builds a TensorRT engine for a Cityscapes segmentation model, serves it with Triton Inference Server, compares Triton with and without dynamic batching, stress-tests local clients, collects metrics, and plots latency-throughput results.

The repository expects:

- ONNX model: `assets/cityscapes_unet.onnx`
- TensorRT plan: `assets/cityscapes_unet.plan`
- Input name: `input`
- Output name: `logits`
- Input shape sent by clients: `[batch, 3, 1024, 2048]`
- Output shape returned by Triton: `[batch, 19, 1024, 2048]`

## Models

Two Triton models are configured from the same TensorRT plan:

- `cityscapes`: baseline without Triton dynamic batching.
- `cityscapes_dyn`: dynamic batching enabled with `max_batch_size: 2`.

For `cityscapes_dyn`, Triton treats the first tensor dimension as the batch dimension because `max_batch_size > 0`. Therefore `models/cityscapes_dyn/config.pbtxt` uses `dims: [ 3, 1024, 2048 ]` for input and `dims: [ 19, 1024, 2048 ]` for output. Clients still send arrays shaped like `[1, 3, 1024, 2048]` or `[2, 3, 1024, 2048]`.

For `cityscapes`, `max_batch_size: 0`, so the config includes the full explicit tensor shape with `dims: [ -1, 3, 1024, 2048 ]`.

## Install Python Client Dependencies

```bash
pip install -r requirements-triton-client.txt
```

## Environment Validation

```bash
sudo docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

This checks that Docker containers can see the GPU through the NVIDIA runtime.

## Build TensorRT Engine

```bash
sudo docker compose run --rm trtexec-build
```

This converts `assets/cityscapes_unet.onnx` into `assets/cityscapes_unet.plan`. The engine profile uses max batch 4 first because the target GPU may have only 8 GB VRAM.

## Benchmark Raw TensorRT Engine

```bash
sudo docker compose run --rm trtexec-benchmark-b1
sudo docker compose run --rm trtexec-benchmark-b2
```

These commands measure raw TensorRT latency and throughput outside Triton for batch sizes 1 and 2.

## Prepare Triton Model Repository

```bash
mkdir -p models/cityscapes/1 models/cityscapes_dyn/1
cp assets/cityscapes_unet.plan models/cityscapes/1/model.plan
cp assets/cityscapes_unet.plan models/cityscapes_dyn/1/model.plan
```

Triton requires model version directories such as `1/`, `2/`, etc. If `assets/cityscapes_unet.plan` does not exist yet, run:

```bash
sudo docker compose run --rm trtexec-build
```

## Run Triton

```bash
sudo docker compose up triton
```

Triton exposes:

- HTTP inference on port `8000`
- gRPC inference on port `8001`
- Prometheus metrics on port `8002`

## Run The Image Pipeline Ensemble

The separate `models_pipeline/` repository serves an ensemble model named `cityscapes_pipeline`. It accepts encoded PNG/JPEG bytes, preprocesses the image in Triton, runs the TensorRT segmentation model, and postprocesses logits into a train-ID mask, a Cityscapes color mask, a class histogram, and mean confidence.

```bash
sudo docker compose up triton-pipeline
```

The pipeline service builds a small image from `docker/Dockerfile.triton-pipeline` to add Pillow for PNG/JPEG decoding in the Python backend. It uses separate ports so it can run independently from the single-model repository:

- HTTP inference on port `8100`
- gRPC inference on port `8101`
- Prometheus metrics on port `8102`

Smoke-test with a real PNG/JPEG:

```bash
python scripts/tests/test_pipeline.py image=path/to/image.png save_color_mask=results/pipeline_color_mask.png
```

Or send a synthetic JPEG when you only want to check the Triton wiring:

```bash
python scripts/tests/test_pipeline.py save_color_mask=results/pipeline_color_mask.png
```

Pipeline metrics:

```bash
curl localhost:8102/metrics | grep cityscapes_pipeline
curl localhost:8102/metrics | grep image_preprocess
curl localhost:8102/metrics | grep cityscapes_embedder
curl localhost:8102/metrics | grep segmentation_postprocess
```

For the ensemble, compare per-step request duration and compute duration to see whether CPU preprocessing/postprocessing or TensorRT execution dominates end-to-end latency.

## Health Checks

```bash
curl -v localhost:8000/v2/health/ready
curl localhost:8000/v2/models/cityscapes
curl localhost:8000/v2/models/cityscapes/config
curl localhost:8000/v2/models/cityscapes_dyn
curl localhost:8000/v2/models/cityscapes_dyn/config
```

## Inference Smoke Tests

Baseline model:

```bash
python scripts/tests/test_simple.py model_name=cityscapes
```

Dynamic-batching model:

```bash
python scripts/tests/test_simple.py model_name=cityscapes_dyn
```

The script sends one random FP32 input and prints input shape, output shape, dtype, min, max, and mean.

## Metrics

```bash
curl localhost:8002/metrics > metrics.txt
curl localhost:8002/metrics | grep cityscapes
curl localhost:8002/metrics | grep nv_inference
```

`metrics.txt` stores raw Prometheus metrics from Triton.

Important metrics:

- `nv_inference_request_success`
- `nv_inference_request_failure`
- `nv_inference_count`
- `nv_inference_exec_count`
- `nv_inference_request_duration_us`
- `nv_inference_queue_duration_us`
- `nv_inference_compute_infer_duration_us`

For the dynamic model, batching is working when backend execution count is lower than successful request count under concurrent load. For example, about 20 successful requests with about 10 backend executions indicates Triton combined requests into batches of 2.

## Load Test

Run one profile manually:

```bash
python scripts/load_test_triton.py --model-name cityscapes --concurrency 4 --batch-size 1 --num-requests 20
python scripts/load_test_triton.py --model-name cityscapes_dyn --concurrency 4 --batch-size 1 --num-requests 20
```

The script writes:

- `results/load_test_raw.csv`
- `results/load_test_summary.csv`

It records timestamp, request id, model name, concurrency, batch size, latency, output shape, success, and error.

## Run Experiments

```bash
bash scripts/run_experiments.sh
```

The runner tests `cityscapes` and `cityscapes_dyn` at concurrency 1, 2, 4, and 8 with batch size 1. It also tests `cityscapes_dyn` with batch size 2 at concurrency 1, 2, and 4. Batch size 2 may be memory-heavy, so failures are recorded and the runner continues.

For each profile, the runner appends a header and a Triton metrics snapshot to `metrics.txt`.

## Plot Results

```bash
python scripts/plot_results.py
```

The plotting script reads `results/load_test_summary.csv` and generates:

- `results/plots/throughput_vs_concurrency.png`
- `results/plots/avg_latency_vs_concurrency.png`
- `results/plots/p95_latency_vs_concurrency.png`
- `results/plots/p99_latency_vs_concurrency.png`
- `results/summary_table.md`

If both models or multiple batch sizes exist in the summary CSV, the plots show separate lines for each model and batch size.

## Perf Analyzer Benchmark

Triton's native Perf Analyzer workflow lives in `perf_analyzer/`.

```bash
sudo docker compose up triton
bash perf_analyzer/run_perf_analyzer.sh
python perf_analyzer/plot_perf_results.py
```

It writes CSV outputs and two plots:

- `perf_analyzer/results/plots/latency_avg_vs_concurrency.png`
- `perf_analyzer/results/plots/throughput_vs_concurrency.png`

## Troubleshooting

- Missing `assets/cityscapes_unet.onnx`: place the ONNX model at that path.
- Missing `assets/cityscapes_unet.plan`: run `sudo docker compose run --rm trtexec-build`.
- GPU out of memory at batch size 2: use batch size 1 or rebuild with smaller TensorRT profiles.
- Triton model config shape mismatch: check whether the request targets `cityscapes` or `cityscapes_dyn`.
- Missing Triton Python client: run `pip install -r requirements-triton-client.txt`.
