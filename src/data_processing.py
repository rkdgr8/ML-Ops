"""Data loading and preprocessing utilities for the UCI Heart Disease dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, UCI_HEART_DISEASE_URL


COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(
    url: str = UCI_HEART_DISEASE_URL,
    output_path: Path = RAW_DATA_PATH,
) -> Path:
    """Download the Cleveland Heart Disease dataset from the UCI repository."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        urlretrieve(url, output_path)

    return output_path


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw UCI Heart Disease dataset with column names."""
    return pd.read_csv(path, names=COLUMN_NAMES, na_values="?")


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and convert the UCI target to binary classification."""
    cleaned = data.copy()

    numeric_columns = ["ca", "thal"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna().reset_index(drop=True)
    cleaned["target"] = (cleaned["target"] > 0).astype(int)

    return cleaned


def save_processed_dataset(data: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    """Save the cleaned dataset for EDA and model training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def prepare_dataset() -> pd.DataFrame:
    """Download, load, clean, and save the Heart Disease dataset."""
    raw_path = download_dataset()
    raw_data = load_dataset(raw_path)
    cleaned_data = clean_dataset(raw_data)
    save_processed_dataset(cleaned_data)
    return cleaned_data


def main() -> None:
    data = prepare_dataset()
    print(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    print("Target distribution:")
    print(data["target"].value_counts().sort_index())


if __name__ == "__main__":
    main()
