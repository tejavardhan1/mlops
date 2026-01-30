"""FastAPI inference: /predict, /health, /metrics. Loads model from MLflow."""
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.inference.model import load_model as load_mlflow_model, get_feature_names

PREDICT_LATENCY = Histogram(
    "mlops_inference_predict_latency_seconds",
    "Predict request latency",
)
PREDICT_TOTAL = Counter(
    "mlops_inference_predict_total",
    "Total predict requests",
    ["status"],
)
PREDICT_ERRORS = Counter(
    "mlops_inference_predict_errors_total",
    "Total predict errors",
)


model = None
feature_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names
    run_id = os.environ.get("MLFLOW_RUN_ID")
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "sklearn_wine")
    try:
        model = load_mlflow_model(run_id=run_id, model_name=model_name)
        feature_names = get_feature_names()
    except Exception as e:
        print(f"Model load failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        model = None
        feature_names = get_feature_names()
    else:
        print("Model loaded from MODEL_PATH", file=sys.stderr, flush=True)
    yield
    model = None


app = FastAPI(title="Mini MLOps Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    probabilities: Optional[List[float]] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    PREDICT_TOTAL.labels(status="received").inc()
    if model is None:
        PREDICT_ERRORS.inc()
        PREDICT_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.features) != len(feature_names):
        PREDICT_ERRORS.inc()
        PREDICT_TOTAL.labels(status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(req.features)}",
        )
    start = time.perf_counter()
    try:
        import numpy as np
        X = np.array([req.features])
        pred = model.predict(X)[0]
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0].tolist()
        PREDICT_LATENCY.observe(time.perf_counter() - start)
        PREDICT_TOTAL.labels(status="success").inc()
        return PredictResponse(prediction=int(pred), probabilities=probs)
    except Exception as e:
        PREDICT_ERRORS.inc()
        PREDICT_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    from src.config import load_config
    cfg = load_config()
    inf = cfg.get("inference", {})
    uvicorn.run(
        "src.inference.app:app",
        host=os.environ.get("INFERENCE_HOST", inf.get("host", "0.0.0.0")),
        port=int(os.environ.get("INFERENCE_PORT", inf.get("port", 8000))),
        reload=False,
    )
