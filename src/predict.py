"""Prediction utilities for saved model artifacts."""

from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model(model_path=MODEL_PATH):
    """Load the saved preprocessing and model pipeline."""
    return joblib.load(model_path)


def normalize_input(input_data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame):
    """Convert supported input formats to a dataframe for sklearn."""
    if isinstance(input_data, pd.DataFrame):
        return input_data

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    return pd.DataFrame(input_data)


def predict(input_data, model_path=MODEL_PATH) -> list[dict]:
    """Run prediction for one or more input records."""
    model = load_model(model_path)
    features = normalize_input(input_data)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return [
        {
            "prediction": int(prediction),
            "confidence": float(probability),
        }
        for prediction, probability in zip(predictions, probabilities, strict=False)
    ]
