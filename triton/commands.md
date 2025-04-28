## Convert to TensorRT

```
sudo docker run -it --rm --gpus '"device=0"' --runtime=nvidia -v ./sources:/models nvcr.io/nvidia/tensorrt:23.12-py3

trtexec --onnx=/models/onnx/conv_model.onnx --saveEngine=/models/tensorrt/conv_model.plan --minShapes=PREPROCESSED_IMAGE:1x3x96x96 --optShapes=PREPROCESSED_IMAGE:16x3x96x96 --maxShapes=PREPROCESSED_IMAGE:64x3x96x96 --profilingVerbosity=detailed --builderOptimizationLevel=5 --fp16
```

## Throughput
```
sudo docker run -it --rm --net=host -v ./perf_logs:/workspace/logs nvcr.io/nvidia/tritonserver:23.12-py3-sdk

perf_analyzer -m conv-encoder-FP16 -u localhost:8900 --concurrency-range 1:8 --shape PREPROCESSED_IMAGE:3,96,96 --measurement-interval 5000 -b 4 > logs/perf_log_batch4.txt
```

