from fastapi import FastAPI


app = FastAPI(title="Heart Disease Prediction API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    return {
        "message": (
            "Prediction endpoint scaffolded. "
            "Model inference will be added later."
        )
    }
