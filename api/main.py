from functools import lru_cache
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import MODEL_PATH


app = FastAPI(title="Heart Disease Prediction API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_api")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions by class",
    ["prediction"],
)


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=0, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    diagnosis: str


@lru_cache
def get_model():
    return joblib.load(MODEL_PATH)


@app.middleware("http")
async def log_and_monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start_time
    endpoint = request.url.path
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(latency)
    logger.info(
        "%s %s completed with %s in %.4fs",
        request.method,
        endpoint,
        status_code,
        latency,
    )

    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    model = get_model()
    feature_values = (
        features.model_dump() if hasattr(features, "model_dump") else features.dict()
    )
    input_data = pd.DataFrame([feature_values])
    prediction = int(model.predict(input_data)[0])
    confidence = float(model.predict_proba(input_data)[0][1])
    diagnosis = "heart disease risk" if prediction == 1 else "no heart disease risk"
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    logger.info(
        "Prediction completed: prediction=%s confidence=%.4f",
        prediction,
        confidence,
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "diagnosis": diagnosis,
    }
