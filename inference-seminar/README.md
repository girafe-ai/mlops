# Inference Formats Seminar

Hands-on seminar on model inference across multiple deployment formats:

- PyTorch eager
- ONNX Runtime on CPU
- ONNX Runtime on NVIDIA CUDA
- ONNX Runtime with TensorRT Execution Provider
- ONNX custom operators
- Boosted trees translated into RAPIDS Forest Inference Library (FIL)

The seminar reuses the Cityscapes segmentation model from
[`cityscapes-segmentation`](../cityscapes-segmentation) and keeps all
runtime settings in Hydra configs.

## Requirements

- Python 3.10+
- NVIDIA GPU for CUDA / TensorRT / FIL GPU parts
- A trained Cityscapes checkpoint compatible with
  `cityscapes-segmentation/cityscapes-segmentation/lightning/model.py`

Install the seminar dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r inference-seminar/requirements.txt
```

If you need Cityscapes data, use the existing downloader:

```bash
python cityscapes-segmentation/scripts/download.py download --destination_path cityscapes-segmentation/data
```

## Layout

| File | Purpose |
|------|---------|
| `01_export_onnx.py` | Export the Cityscapes model to ONNX and validate parity |
| `02_benchmark_backends.py` | Benchmark PyTorch, ORT CPU, ORT CUDA, ORT TensorRT |
| `03_custom_op_demo.py` | Build an ONNX graph with a custom operator |
| `04_fil_benchmark.py` | Compare XGBoost inference with RAPIDS FIL |
| `SUMMARY.md` | What to conclude from the measurements |
| `conf/config.yaml` | Main Hydra entry point |
| `results/` | Default output directory for artifacts and CSVs |

## Quick Start

Export ONNX:

```bash
python inference-seminar/01_export_onnx.py \
  paths.cityscapes_checkpoint=/abs/path/to/best.ckpt
```

Run the backend benchmark:

```bash
python inference-seminar/02_benchmark_backends.py \
  paths.cityscapes_checkpoint=/abs/path/to/best.ckpt
```

Run the custom op demo:

```bash
python inference-seminar/03_custom_op_demo.py
```

Run the FIL benchmark:

```bash
python inference-seminar/04_fil_benchmark.py
```

## Notes

- The TensorRT benchmark is implemented through the ONNX Runtime TensorRT
  Execution Provider. This keeps the seminar focused on format/runtime
  comparisons while still exercising NVIDIA TensorRT underneath.
- If a provider is unavailable on the current machine, the benchmark records
  that fact instead of crashing the entire seminar.
