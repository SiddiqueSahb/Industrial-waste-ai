"""WaRP-C inference service — FastAPI.

Serves a single trained classifier (chosen via env vars) with:
    GET  /health   liveness + model info
    POST /predict  image upload -> predicted class + confidence (+ top-k)

Run locally from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Configuration (environment variables):
    MODEL_NAME       architecture from configs/classification.yaml (default: convnext_tiny)
    WEIGHTS_PATH     checkpoint path (default: models/classification/<weights_file>)
    DEVICE           cuda / mps / cpu (default: auto-detect)
    TOP_K            entries returned in top_k (default: 5)
    MODEL_URI        gs://bucket/path.pth or https:// URL — downloaded to
                     WEIGHTS_PATH at startup when the checkpoint is missing
                     (how Cloud Run pulls weights from the GCS model registry)
    ALLOW_UNTRAINED  "1" = serve randomly-initialised weights if the checkpoint
                     is missing. For CI smoke tests only — never in production.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import HealthResponse, PredictionResponse, TopKEntry
from src.classification.augmentations import get_eval_transform
from src.classification.inference import load_trained_model, predict_image
from src.classification.models import get_model
from src.config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("warp-api")

MODEL_NAME = os.environ.get("MODEL_NAME", "convnext_tiny")
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(ROOT / "configs" / "classification.yaml"))
TOP_K = int(os.environ.get("TOP_K", "5"))
ALLOW_UNTRAINED = os.environ.get("ALLOW_UNTRAINED", "0") == "1"

state: dict = {}


def _pick_device():
    import torch
    requested = os.environ.get("DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fetch_weights(uri: str, dest: Path) -> None:
    """Download a checkpoint from gs:// or https:// into `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading weights from %s ...", uri)
    if uri.startswith("gs://"):
        from google.cloud import storage  # optional dep, only needed for gs://
        bucket_name, _, blob_path = uri[len("gs://"):].partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).download_to_filename(str(dest))
    elif uri.startswith(("http://", "https://")):
        import urllib.request
        urllib.request.urlretrieve(uri, str(dest))
    else:
        raise ValueError(f"Unsupported MODEL_URI scheme: {uri}")
    log.info("Weights downloaded to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(CONFIG_PATH)
    if MODEL_NAME not in cfg.models:
        raise RuntimeError(f"MODEL_NAME={MODEL_NAME!r} not in config. Available: {list(cfg.models)}")
    mc = cfg.models[MODEL_NAME]

    weights = Path(os.environ.get(
        "WEIGHTS_PATH",
        ROOT / "models" / "classification" / (mc.weights_file or f"{MODEL_NAME}_best.pth"),
    ))
    device = _pick_device()

    # LFS pointer files are a few hundred bytes — treat them as missing.
    weights_ok = weights.exists() and weights.stat().st_size > 10_000
    model_uri = os.environ.get("MODEL_URI")
    if not weights_ok and model_uri:
        _fetch_weights(model_uri, weights)
        weights_ok = weights.exists() and weights.stat().st_size > 10_000
    if weights_ok:
        log.info("Loading %s weights from %s on %s", MODEL_NAME, weights, device.type)
        model = load_trained_model(MODEL_NAME, weights, num_classes=cfg.num_classes,
                                   dropout=mc.dropout, device=device)
    elif ALLOW_UNTRAINED:
        log.warning("Checkpoint %s missing — serving UNTRAINED %s (ALLOW_UNTRAINED=1)",
                    weights, MODEL_NAME)
        model = get_model(MODEL_NAME, num_classes=cfg.num_classes,
                          pretrained=False, dropout=mc.dropout)
        model.to(device).eval()
    else:
        raise RuntimeError(
            f"Checkpoint not found or invalid: {weights}. "
            "Train the model or set WEIGHTS_PATH (or ALLOW_UNTRAINED=1 for smoke tests)."
        )

    state["model"] = model
    state["classes"] = cfg.classes
    state["device"] = device.type
    state["weights_loaded"] = weights_ok
    state["transform"] = get_eval_transform(
        img_size=mc.image_size,
        resize=mc.raw.get("resize_for_eval", cfg.augmentation.resize_for_eval),
        mean=cfg.augmentation.mean,
        std=cfg.augmentation.std,
    )
    log.info("Model ready: %s (%d classes, weights_loaded=%s)",
             MODEL_NAME, cfg.num_classes, weights_ok)
    yield
    state.clear()


app = FastAPI(
    title="WaRP-C Industrial Waste Classifier",
    description="Classifies waste images into 28 WaRP-C recyclable categories.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        device=state["device"],
        weights_loaded=state["weights_loaded"],
        num_classes=len(state["classes"]),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    t0 = time.perf_counter()
    probs = predict_image(state["model"], image, state["transform"])
    latency_ms = (time.perf_counter() - t0) * 1000

    classes = state["classes"]
    order = probs.argsort()[::-1][:TOP_K]
    top_k = [TopKEntry(class_name=classes[i], confidence=round(float(probs[i]), 6))
             for i in order]

    return PredictionResponse(
        model_name=MODEL_NAME,
        predicted_class=top_k[0].class_name,
        confidence=top_k[0].confidence,
        top_k=top_k,
        latency_ms=round(latency_ms, 2),
    )
