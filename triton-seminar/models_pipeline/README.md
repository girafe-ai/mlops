# Triton Ensemble Pipeline

This model repository exposes an image-to-mask Triton ensemble:

1. `image_preprocess`
   - Python backend.
   - Input: encoded PNG/JPEG bytes as `IMAGE_BYTES`.
   - Output: normalized NCHW FP32 tensor named `IMAGE_TENSOR`.
   - Mirrors the Cityscapes repo validation preprocessing: RGB conversion, resize to `1024x2048`, ImageNet mean/std normalization.

2. `cityscapes_embedder`
   - TensorRT plan model.
   - Input: `input`.
   - Output: `logits`.
   - Uses `assets/cityscapes_unet.plan`, copied to `models_pipeline/cityscapes_embedder/1/model.plan`.

3. `segmentation_postprocess`
   - Python backend.
   - Input: `LOGITS`.
   - Outputs:
     - `PRED_MASK`: train ID mask, shape `[1, 1024, 2048]`.
     - `COLOR_MASK`: Cityscapes RGB mask, shape `[1, 1024, 2048, 3]`.
     - `CLASS_HISTOGRAM`: pixel counts per class, shape `[19]`.
     - `MEAN_CONFIDENCE`: mean max softmax probability, shape `[1]`.

The public ensemble model is `cityscapes_pipeline`.

## Run

```bash
sudo docker compose up triton-pipeline
```

The pipeline service builds `docker/Dockerfile.triton-pipeline`, which adds Pillow to Triton's Python environment for PNG/JPEG decoding. It uses separate ports:

- HTTP: `8100`
- gRPC: `8101`
- Metrics: `8102`

## Smoke Test

With a real image:

```bash
python scripts/tests/test_pipeline.py image=path/to/image.png save_color_mask=results/pipeline_color_mask.png
```

Without an image, the script sends a synthetic JPEG:

```bash
python scripts/tests/test_pipeline.py save_color_mask=results/pipeline_color_mask.png
```

## Metrics

```bash
curl localhost:8102/metrics | grep cityscapes_pipeline
curl localhost:8102/metrics | grep image_preprocess
curl localhost:8102/metrics | grep cityscapes_embedder
curl localhost:8102/metrics | grep segmentation_postprocess
```

Useful metrics:

- `nv_inference_request_success`
- `nv_inference_request_failure`
- `nv_inference_request_duration_us`
- `nv_inference_queue_duration_us`
- `nv_inference_compute_infer_duration_us`
- `nv_inference_exec_count`

For this ensemble, compare per-step timings to see whether CPU preprocessing/postprocessing or TensorRT execution dominates end-to-end latency.

## Recommended Next Additions

- Return `ORIGINAL_SHAPE` from preprocessing and resize `PRED_MASK`/`COLOR_MASK` back to the source image size in postprocessing.
- Add an overlay output that blends the original image and segmentation mask for easier visual debugging.
- Add a pipeline load-test script that sends encoded image bytes at concurrency 1, 2, 4, and 8, then compares per-step metrics.
- Add an evaluation script that accepts Cityscapes ground-truth masks and computes mIoU from `PRED_MASK`.
- Add a dynamic-batching pipeline variant after the byte-input baseline works. Variable encoded image sizes are awkward to batch, so the clean route is usually to batch after preprocessing.
