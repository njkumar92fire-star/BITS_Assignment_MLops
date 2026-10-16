"""FastAPI service exposing the heart-disease model.

Endpoints
  GET  /health   liveness / readiness
  GET  /info     model metadata
  POST /predict  JSON in -> prediction + confidence out
  GET  /metrics  Prometheus metrics
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
)

from api.schemas import PatientFeatures, PredictionResponse
from src.predict import load_metadata, load_model, predict_records

LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("heart-api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")  # messages are JSON lines
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=2_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:  # read-only filesystem: stdout logging is enough
        pass
    logger.propagate = False
    return logger


log = configure_logging()

# A private registry keeps metrics clean when the module is imported by tests.
REGISTRY = CollectorRegistry()
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("heart_api_request_duration_seconds", "Request latency (s)",
                    ["endpoint"], registry=REGISTRY,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served",
                      ["prediction"], registry=REGISTRY)
CONFIDENCE = Histogram("heart_api_prediction_confidence", "Confidence of served predictions",
                       registry=REGISTRY, buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
ERRORS = Counter("heart_api_errors_total", "Unhandled errors", registry=REGISTRY)
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded", registry=REGISTRY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.metadata = {}
    try:
        app.state.model = load_model()
        app.state.metadata = load_metadata()
        MODEL_LOADED.set(1)
        log.info(json.dumps({"event": "model_loaded",
                             "model": app.state.metadata.get("model_name", "unknown")}))
    except Exception as exc:
        MODEL_LOADED.set(0)
        log.error(json.dumps({"event": "model_load_failed", "error": str(exc)}))
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        ERRORS.inc()
        log.exception(json.dumps({"event": "unhandled_error", "request_id": request_id}))
        raise
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        if endpoint != "/metrics":
            log.info(json.dumps({
                "event": "request", "request_id": request_id, "method": request.method,
                "path": request.url.path, "status": status,
                "latency_ms": round(elapsed * 1000, 2),
            }))


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.get("/info")
def info(request: Request):
    meta = request.app.state.metadata
    return {
        "model_name": meta.get("model_name"),
        "trained_at": meta.get("trained_at"),
        "sklearn_version": meta.get("sklearn_version"),
        "test_metrics": meta.get("test_metrics"),
        "features": meta.get("feature_columns"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    result = predict_records(model, [patient.model_dump()])[0]
    PREDICTIONS.labels(str(result["prediction"])).inc()
    CONFIDENCE.observe(result["confidence"])
    log.info(json.dumps({
        "event": "prediction", "request_id": request.state.request_id,
        "prediction": result["prediction"], "confidence": result["confidence"],
    }))
    return {**result,
            "model_name": request.app.state.metadata.get("model_name", "unknown"),
            "request_id": request.state.request_id}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
