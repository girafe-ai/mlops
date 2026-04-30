You are working in an MLOps seminar repository for Triton Inference Server + TensorRT.

Goal:
Create a reproducible seminar setup that shows:
1. Building a TensorRT engine with trtexec.
2. Serving it with Triton Inference Server.
3. Enabling Triton dynamic batching.
4. Stress-testing Triton with several local clients.
5. Collecting metrics.
6. Plotting throughput/latency versus concurrency.
7. Writing conclusions.

Assumptions:
- The repository already contains or will contain an ONNX model at:

  assets/cityscapes_unet.onnx

- The model input name is:

  input

- The model output name is:

  logits

- The model is a Cityscapes segmentation model.
- Input shape is NCHW:

  [batch, 3, 1024, 2048]

- Output shape is:

  [batch, 19, 1024, 2048]

- Docker GPU access on this machine works with:

  runtime: nvidia

  not necessarily with:

  --gpus all

- Use Docker Compose for all Docker-based steps.

Tasks:

1. Create or verify docker-compose.yml

Create docker-compose.yml if it does not exist.
If it already exists, inspect it and update it carefully without breaking existing useful services.

The compose file must contain at least these services:

A. trtexec-build

Purpose:
Build TensorRT engine from ONNX.

Image:
nvcr.io/nvidia/tensorrt:24.02-py3

Requirements:
- runtime: nvidia
- NVIDIA_VISIBLE_DEVICES=all
- NVIDIA_DRIVER_CAPABILITIES=compute,utility
- mount repo root to /workspace
- working_dir: /workspace
- ipc: host
- ulimits:
  - memlock: -1
  - stack: 67108864

Command should be similar to:

trtexec
--onnx=assets/cityscapes_unet.onnx
--saveEngine=assets/cityscapes_unet.plan
--fp16
--minShapes=input:1x3x1024x2048
--optShapes=input:1x3x1024x2048
--maxShapes=input:2x3x1024x2048
--verbose

Use max batch 2 first because the target GPU may have only 8 GB VRAM.
Do not default to batch 4 unless the user explicitly asks.

B. trtexec-benchmark-b1

Purpose:
Benchmark TensorRT engine with batch size 1.

Command:

trtexec
--loadEngine=assets/cityscapes_unet.plan
--shapes=input:1x3x1024x2048
--warmUp=500
--duration=30
--useSpinWait

C. trtexec-benchmark-b2

Purpose:
Benchmark TensorRT engine with batch size 2.

Command:

trtexec
--loadEngine=assets/cityscapes_unet.plan
--shapes=input:2x3x1024x2048
--warmUp=500
--duration=30
--useSpinWait

D. triton

Purpose:
Serve TensorRT engine using Triton Inference Server.

Image:
nvcr.io/nvidia/tritonserver:24.02-py3

Requirements:
- runtime: nvidia
- NVIDIA_VISIBLE_DEVICES=all
- NVIDIA_DRIVER_CAPABILITIES=compute,utility
- expose ports:
  - 8000:8000 HTTP
  - 8001:8001 gRPC
  - 8002:8002 metrics
- mount ./models:/models

Command:

tritonserver
--model-repository=/models
--log-verbose=1

2. Create or verify Triton model repository

Create the following structure if missing:

models/
└── cityscapes/
    ├── config.pbtxt
    └── 1/
        └── model.plan

If assets/cityscapes_unet.plan exists, copy it to:

models/cityscapes/1/model.plan

If it does not exist, do not fail silently.
Add a clear README instruction telling the user to run:

sudo docker compose run --rm trtexec-build

before starting Triton.

3. Create or verify dynamic batching config

Create or update:

models/cityscapes/config.pbtxt

Use Triton-managed batching config:

name: "cityscapes"
platform: "tensorrt_plan"

max_batch_size: 2

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 1024, 2048 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 19, 1024, 2048 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 2 ]
  max_queue_delay_microseconds: 10000
}

Important:
Explain in README.md that because max_batch_size > 0, Triton treats the first tensor dimension as the batch dimension. Therefore dims exclude batch.

The client should still send arrays with shape:

[batch, 3, 1024, 2048]

For example:

[1, 3, 1024, 2048]

4. Create README.md

Create or update README.md with a clear seminar-style walkthrough.

The README must include:

A. Environment validation

Commands:

sudo docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

Explain:
This checks that Docker containers can see the GPU.

B. Build TensorRT engine

Command:

sudo docker compose run --rm trtexec-build

Explain:
This converts assets/cityscapes_unet.onnx into assets/cityscapes_unet.plan.

C. Benchmark raw TensorRT engine

Commands:

sudo docker compose run --rm trtexec-benchmark-b1
sudo docker compose run --rm trtexec-benchmark-b2

Explain:
This measures raw TensorRT latency and throughput outside Triton.

D. Prepare Triton model repository

Commands:

mkdir -p models/cityscapes/1
cp assets/cityscapes_unet.plan models/cityscapes/1/model.plan

Explain:
Triton requires model version directories like 1/, 2/, etc.

E. Run Triton

Command:

sudo docker compose up triton

Explain:
Triton exposes:
- HTTP inference on 8000
- gRPC inference on 8001
- Prometheus metrics on 8002

F. Health checks

Commands:

curl -v localhost:8000/v2/health/ready
curl localhost:8000/v2/models/cityscapes
curl localhost:8000/v2/models/cityscapes/config

G. Metrics

Commands:

curl localhost:8002/metrics > metrics.txt
curl localhost:8002/metrics | grep cityscapes
curl localhost:8002/metrics | grep nv_inference

Explain:
metrics.txt stores raw Prometheus metrics from Triton.

H. Load test

Document how to run the load-test script that you will create.

I. Plot results

Document how to run the plotting script that you will create.

5. Create inference smoke test script

Create:

scripts/infer_triton_random.py

It should:
- Use tritonclient[http]
- Send one random input with shape [1, 3, 1024, 2048]
- Request output logits
- Print:
  - input shape
  - output shape
  - output dtype
  - min/max/mean of output

Use model name:

cityscapes

Use URL:

localhost:8000

6. Create load-testing script

Create:

scripts/load_test_triton.py

The script must:
- Use concurrent local clients from localhost.
- Use HTTP Triton client.
- Accept CLI arguments:
  - --url, default localhost:8000
  - --model-name, default cityscapes
  - --height, default 1024
  - --width, default 2048
  - --concurrency, default 1
  - --num-requests, default 20
  - --batch-size, default 1
  - --output-csv, default results/load_test_results.csv

Behavior:
- Spawn N workers according to --concurrency.
- Each request sends random FP32 input of shape:

  [batch_size, 3, height, width]

- Measure client-side latency per request with time.perf_counter().
- Record at least:
  - timestamp
  - request_id
  - concurrency
  - batch_size
  - latency_ms
  - output_shape
  - success
  - error, if failed

At the end, append summary rows or write a separate summary CSV with:
- concurrency
- batch_size
- num_requests
- successful_requests
- failed_requests
- total_time_s
- throughput_rps
- avg_latency_ms
- p50_latency_ms
- p90_latency_ms
- p95_latency_ms
- p99_latency_ms

Prefer creating:

results/load_test_raw.csv
results/load_test_summary.csv

The script should create results/ if it does not exist.

7. Create experiment runner

Create:

scripts/run_experiments.sh

It should run several load profiles, for example:

concurrency = 1, 2, 4, 8
batch_size = 1

Optionally also test:

concurrency = 1, 2, 4
batch_size = 2

But batch size 2 may be memory-heavy, so handle failure gracefully.

For each experiment:
- Run scripts/load_test_triton.py
- Append results to CSV
- Fetch Triton metrics and append to metrics.txt

Use commands similar to:

curl localhost:8002/metrics >> metrics.txt

Before appending metrics for each experiment, write a header into metrics.txt, for example:

===== concurrency=4 batch_size=1 timestamp=... =====

8. Create plotting script

Create:

scripts/plot_results.py

It must:
- Read results/load_test_summary.csv
- Generate plots into:

results/plots/

Required plots:
- throughput_rps vs concurrency
- avg_latency_ms vs concurrency
- p95_latency_ms vs concurrency
- p99_latency_ms vs concurrency

If multiple batch sizes exist, plot separate lines for each batch size.

Use matplotlib.
Do not require seaborn.

Also save a Markdown-friendly summary table to:

results/summary_table.md

9. Collect and save metrics

Create or update:

metrics.txt

This file should contain:
- raw Triton Prometheus metrics snapshots
- clearly separated by experiment profile
- include concurrency and batch size in section headers

Important metrics to inspect and mention in conclusions:
- nv_inference_request_success
- nv_inference_request_failure
- nv_inference_count
- nv_inference_exec_count
- nv_inference_request_duration_us
- nv_inference_queue_duration_us
- nv_inference_compute_infer_duration_us

The key dynamic batching check:
If dynamic batching is working, backend execution count should be lower than the number of successful client requests under concurrent load.

For example:
- request_success ≈ 20
- exec_count ≈ 10

This indicates Triton combined requests into batches of 2.

10. Create CONCLUSIONS.md

Create:

CONCLUSIONS.md

It must include:

A. What was tested

Explain:
- TensorRT engine built from ONNX.
- Triton served TensorRT plan.
- Dynamic batching enabled.
- Local concurrent clients generated load.
- Metrics were collected from Triton Prometheus endpoint.

B. Key observations

Use actual values from results/load_test_summary.csv when available.

Discuss:
- how throughput changes with concurrency
- how average latency changes with concurrency
- how p95/p99 latency changes with concurrency
- whether dynamic batching appears to be working
- whether queue duration increases at high concurrency
- whether GPU/compute duration dominates latency

C. Latency-throughput tradeoff

Explain:
- low concurrency usually gives lower latency
- higher concurrency may improve throughput
- too much concurrency increases queueing and tail latency

D. Triton dynamic batching conclusion

Explain:
Dynamic batching helps when multiple small requests arrive close together.
It improves throughput by combining requests, but may add queue delay.

E. Recommendations

Include practical recommendations:
- Start with max_batch_size 2 for 8 GB GPUs.
- Tune max_queue_delay_microseconds carefully.
- Use low delay for online latency-sensitive inference.
- Use higher delay for throughput-oriented workloads.
- Watch p95/p99 latency, not only average latency.
- Monitor exec_count vs request_count to verify batching.

11. Check all files

After creating files, run checks:

A. YAML syntax check if Python yaml is available:

python - <<'PY'
import yaml
with open("docker-compose.yml") as f:
    yaml.safe_load(f)
print("docker-compose.yml OK")
PY

If PyYAML is not installed, skip gracefully and mention it.

B. Check required files exist:

- docker-compose.yml
- README.md
- models/cityscapes/config.pbtxt
- scripts/infer_triton_random.py
- scripts/load_test_triton.py
- scripts/run_experiments.sh
- scripts/plot_results.py
- CONCLUSIONS.md

C. Check Compose config:

sudo docker compose config

D. Check Triton config consistency:
- max_batch_size must be 2
- input dims must be [3, 1024, 2048], not [1, 3, 1024, 2048]
- output dims must be [19, 1024, 2048], not [1, 19, 1024, 2048]
- model name must be cityscapes
- platform must be tensorrt_plan

12. Do not hide failures

If any command fails:
- Capture the exact command.
- Capture the error.
- Write a short troubleshooting note in README.md or CONCLUSIONS.md.

Common expected failures:
- assets/cityscapes_unet.onnx missing
- assets/cityscapes_unet.plan missing
- GPU out of memory for batch size 2
- Triton model config shape mismatch
- Triton client package missing

13. Dependency notes

If creating a requirements file is useful, create:

requirements-triton-client.txt

with:

numpy
tritonclient[http]
matplotlib
pandas

Do not install dependencies automatically unless the environment is intended for it.
Document installation command:

pip install -r requirements-triton-client.txt

14. Final deliverables

At the end, the repository should contain:

docker-compose.yml
README.md
CONCLUSIONS.md
metrics.txt
requirements-triton-client.txt
models/cityscapes/config.pbtxt
scripts/infer_triton_random.py
scripts/load_test_triton.py
scripts/run_experiments.sh
scripts/plot_results.py
results/load_test_raw.csv
results/load_test_summary.csv
results/summary_table.md
results/plots/throughput_vs_concurrency.png
results/plots/avg_latency_vs_concurrency.png
results/plots/p95_latency_vs_concurrency.png
results/plots/p99_latency_vs_concurrency.png

If some runtime outputs cannot be generated because Triton is not running or the engine/model is missing, create all scripts and documentation anyway, and clearly state what command the user must run next.
