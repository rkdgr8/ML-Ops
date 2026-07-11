from fastapi.testclient import TestClient

import api.main as api_main


class FakeModel:
    def predict(self, input_data):
        return [1]

    def predict_proba(self, input_data):
        return [[0.18, 0.82]]


def test_predict_returns_prediction(monkeypatch):
    api_main.get_model.cache_clear()
    monkeypatch.setattr(api_main, "get_model", lambda: FakeModel())

    client = TestClient(api_main.app)
    response = client.post(
        "/predict",
        json={
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "confidence": 0.82,
        "diagnosis": "heart disease risk",
    }
