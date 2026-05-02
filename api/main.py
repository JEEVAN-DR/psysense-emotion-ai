"""
PsySense Emotion AI — FastAPI inference service.

Endpoints
---------
GET  /health         — liveness / readiness probe
GET  /metrics        — Prometheus metrics (via prometheus-fastapi-instrumentator)
POST /predict        — single-text emotion classification
POST /predict/batch  — batch emotion classification (up to 32 texts)
GET  /docs           — Swagger UI (automatic)
GET  /redoc          — ReDoc (automatic)
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.dependencies import get_model_bundle
from api.models import (
    EmotionScore,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from config import settings
from inference import predict_emotions

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level.upper(),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("psysense.api")

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model at startup so the first request is not slow."""
    logger.info("Starting PsySense Emotion AI API (env=%s) …", settings.environment)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_model_bundle)
    logger.info("Model warm-up complete.")
    yield
    logger.info("Shutting down PsySense Emotion AI API.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PsySense Emotion AI API",
    description=(
        "Production-grade inference API for multi-label emotion classification "
        "using a DistilBERT model fine-tuned on the GoEmotions dataset."
    ),
    version="1.0.0",
    contact={
        "name": "Hitan K",
        "url": "https://github.com/Hitan547/psysense-emotion-ai",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Prometheus metrics ────────────────────────────────────────────────────────

if settings.metrics_enabled:
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint=settings.metrics_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_prediction(request: PredictRequest) -> PredictResponse:
    """Execute prediction synchronously (runs in a thread-pool executor)."""
    model, tokenizer, mlb, device = get_model_bundle()
    label_names = list(mlb.classes_)

    t0 = time.perf_counter()
    result = predict_emotions(
        model,
        tokenizer,
        label_names,
        device,
        request.text,
        threshold=request.threshold,
        top_k=request.top_k,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return PredictResponse(
        text=request.text,
        dominant_emotion=EmotionScore(**result["dominant_emotion"]),
        active_emotions=[EmotionScore(**e) for e in result["active_emotions"]],
        top_emotions=[
            EmotionScore(label=label, confidence=conf)
            for label, conf in result["top_emotions"]
        ],
        processing_time_ms=round(elapsed_ms, 2),
        model_version=settings.hf_model,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness / readiness probe for load-balancers and k8s."""
    try:
        bundle = get_model_bundle()
        model_loaded = bundle is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        environment=settings.environment,
        version=app.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    """
    Classify emotions in a single text string.

    Returns the dominant emotion, all active emotions above the threshold,
    and a ranked top-k list with confidence scores.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_prediction, request)


@app.post("/predict/batch", response_model=List[PredictResponse], tags=["inference"])
async def predict_batch(requests: List[PredictRequest]):
    """
    Classify emotions for a batch of up to 32 texts concurrently.
    """
    if len(requests) > 32:
        raise HTTPException(
            status_code=422,
            detail="Batch size must not exceed 32 items.",
        )
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _run_prediction, r) for r in requests]
    return await asyncio.gather(*tasks)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
