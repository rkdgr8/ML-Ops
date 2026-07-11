from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import MODEL_PATH


app = FastAPI(title="Heart Disease Prediction API")


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


@app.get("/health")
def health_check():
    return {"status": "ok"}


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

    return {
        "prediction": prediction,
        "confidence": confidence,
        "diagnosis": diagnosis,
    }
