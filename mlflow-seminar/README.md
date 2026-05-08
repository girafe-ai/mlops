# MLflow Serving

## Setup

Install dependencies in your preferred environment:

```bash
pip install -r requirements.txt
```

Place the trained checkpoint at the default path:

```text
../cityscapes-segmentation/checkpoints/best.ckpt
```

or override `register.checkpoint` when registering.

## Register The Existing Model

From this directory:

```bash
python register_model.py register.checkpoint=../cityscapes-segmentation/checkpoints/best.ckpt
```

The default registration config lives in `conf/config.yaml` and
`conf/register/default.yaml`.

This creates or updates:

- experiment: `mlflow-cityscapes-serving`
- registered model: `cityscapes-segmentation`
- alias: `champion`

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Serve The Registered Model

```bash
bash scripts/serve_model.sh
```

The served endpoint is:

```text
POST http://127.0.0.1:5001/invocations
```

Smoke test it:

```bash
python test_serving.py
```

Use Hydra overrides for another image or endpoint:

```bash
python test_serving.py test.image=assets/cityscapes_example.jpeg test.url=http://127.0.0.1:5001/invocations
```

For a local-only demo, the served model also accepts an image path:

```bash
python test_serving.py test.input_mode=path test.image=assets/cityscapes_example.jpeg
```

The same request with `curl`:

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"image_path":"/absolute/path/to/mlflow-seminar/assets/cityscapes_example.jpeg"}]}'
```

## Run The Website

Keep MLflow serving running, then start the web app in another terminal:

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Upload `assets/cityscapes_example.jpeg` or any street-scene image.

## Teaching Notes

MLflow serving accepts JSON, while browsers upload files as multipart form data.
The FastAPI app bridges that gap by converting the image to base64 and sending:

```json
{
  "dataframe_records": [
    {
      "image_base64": "..."
    }
  ]
}
```

For local debugging, `image_path` is convenient. For a real remote service,
prefer `image_base64`: the model server usually cannot read files from the
client machine.

The model returns display-ready PNGs:

```json
{
  "mask_base64": "...",
  "overlay_base64": "...",
  "mean_confidence": 0.91,
  "class_histogram": {
    "road": {
      "pixels": 1000,
      "share": 0.31
    }
  }
}
```

The website targets the served model URI, while
deployment decisions live in the registry alias:

```text
models:/cityscapes-segmentation@champion
```
