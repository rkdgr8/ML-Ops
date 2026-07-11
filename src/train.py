"""Model training and evaluation entry point."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_METRICS_PATH, MODEL_OUTPUT_DIR, MODEL_PATH
from src.eda import load_or_prepare_processed_data


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataset into features and target."""
    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_searches() -> dict[str, GridSearchCV]:
    """Create tuned model candidates for comparison."""
    preprocessor = build_preprocessor()

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return {
        "logistic_regression": GridSearchCV(
            logistic_pipeline,
            param_grid={
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
            },
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            forest_pipeline,
            param_grid={
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 4, 8],
                "classifier__min_samples_split": [2, 5],
            },
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted model on the test set."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def save_confusion_matrix(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = MODEL_OUTPUT_DIR,
) -> None:
    """Save a confusion matrix plot for the final model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=["No Disease", "Disease"],
        cmap="Blues",
    )
    display.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def save_roc_curve(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = MODEL_OUTPUT_DIR,
) -> None:
    """Save an ROC curve plot for the final model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    sns.lineplot(
        x=false_positive_rate,
        y=true_positive_rate,
        label=f"AUC = {auc_score:.3f}",
    )
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()


def save_training_outputs(
    model: Pipeline,
    metrics: dict,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = MODEL_METRICS_PATH,
) -> None:
    """Save the final model pipeline and metrics summary."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_models(data: pd.DataFrame) -> tuple[str, Pipeline, dict]:
    """Train candidate models and return the best one by ROC-AUC."""
    x, y = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_results = {}
    best_name = ""
    best_model = None
    best_roc_auc = -1.0

    for name, search in build_model_searches().items():
        search.fit(x_train, y_train)
        model = search.best_estimator_
        test_metrics = evaluate_model(model, x_test, y_test)
        cross_val_auc = cross_val_score(
            model,
            x_train,
            y_train,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )

        model_results[name] = {
            "best_params": search.best_params_,
            "cross_val_roc_auc_mean": cross_val_auc.mean(),
            "cross_val_roc_auc_std": cross_val_auc.std(),
            "test_metrics": test_metrics,
        }

        if test_metrics["roc_auc"] > best_roc_auc:
            best_name = name
            best_model = model
            best_roc_auc = test_metrics["roc_auc"]

    if best_model is None:
        raise RuntimeError("No model was trained.")

    summary = {
        "best_model": best_name,
        "models": model_results,
    }

    save_confusion_matrix(best_model, x_test, y_test)
    save_roc_curve(best_model, x_test, y_test)
    save_training_outputs(best_model, summary)

    return best_name, best_model, summary


def main() -> None:
    """Train and evaluate candidate heart disease models."""
    data = load_or_prepare_processed_data()
    best_name, _, summary = train_models(data)

    print(f"Best model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {MODEL_METRICS_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
