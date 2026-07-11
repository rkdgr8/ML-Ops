from unittest.mock import MagicMock

from src.train import log_experiment_runs


def test_log_experiment_runs_logs_each_model(monkeypatch):
    logged_runs = []

    def fake_log_model_run(model_name, model, result, is_best):
        logged_runs.append(
            {
                "model_name": model_name,
                "model": model,
                "result": result,
                "is_best": is_best,
            }
        )

    monkeypatch.setattr("src.train.mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("src.train.log_model_run", fake_log_model_run)

    fitted_models = {
        "logistic_regression": object(),
        "random_forest": object(),
    }
    summary = {
        "models": {
            "logistic_regression": {"roc_auc": 0.9},
            "random_forest": {"roc_auc": 0.8},
        }
    }

    log_experiment_runs(fitted_models, summary, best_name="logistic_regression")

    assert len(logged_runs) == 2
    assert logged_runs[0]["is_best"] is True
    assert logged_runs[1]["is_best"] is False
