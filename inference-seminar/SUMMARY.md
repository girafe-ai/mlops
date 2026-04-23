# Summary — inference format comparison

## What students should see

- **PyTorch eager** is the easiest reference path and the right baseline for
  correctness checks.
- **ONNX Runtime CPU** is usually the most portable deployment format.
- **ONNX Runtime CUDA** removes Python overhead and can outperform eager
  inference on GPU.
- **ONNX Runtime TensorRT EP** often delivers the best steady-state GPU
  latency, but engine build time and shape constraints matter.
- **Custom operators** are possible in the ONNX ecosystem, but they create an
  extra portability constraint because the runtime must know how to execute
  that op.
- **Boosting -> FIL** is a strong example of format translation beyond neural
  networks: the same trained ensemble can be served by a specialized inference
  engine.

## Suggested discussion points

- Compare not only average latency, but also:
  - warmup cost
  - p50 / p95 latency
  - throughput at batch size > 1
  - numerical parity against the PyTorch reference
- TensorRT is strongest when:
  - input shapes are stable
  - deployment is GPU-only
  - paying engine-build cost once is acceptable
- ONNX custom ops are useful when:
  - pre/post-processing must stay inside one graph
  - the operation is domain-specific
  - the runtime environment is controlled

## Recommended order in class

1. Export the segmentation model to ONNX.
2. Validate that ONNX outputs stay close to PyTorch outputs.
3. Benchmark ORT CPU, ORT CUDA, and ORT TensorRT EP.
4. Show a small custom-operator graph in ONNX Runtime.
5. Switch to boosted trees and translate them to FIL.
