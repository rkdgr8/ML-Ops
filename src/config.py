from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

UCI_HEART_DISEASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DATA_PATH = DATA_DIR / "raw" / "heart.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_processed.csv"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
EDA_OUTPUT_DIR = SCREENSHOTS_DIR / "eda"
