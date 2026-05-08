"""Small website backend for Cityscapes segmentation via MLflow Serving."""

from __future__ import annotations

import base64
import os

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_DIR, "static")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5001/invocations")

app = FastAPI(title="Cityscapes MLflow Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    response = requests.post(
        MLFLOW_URL,
        json={"dataframe_records": [{"image_base64": image_base64}]},
        timeout=180,
    )
    response.raise_for_status()
    prediction = response.json()["predictions"][0]
    return {
        "mask": f"data:image/png;base64,{prediction['mask_base64']}",
        "overlay": f"data:image/png;base64,{prediction['overlay_base64']}",
        "mean_confidence": prediction["mean_confidence"],
        "class_histogram": prediction["class_histogram"],
    }
