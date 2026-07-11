import pandas as pd

from src.data_processing import clean_dataset


def test_clean_dataset_converts_target_to_binary():
    data = pd.DataFrame(
        {
            "age": [63, 67],
            "sex": [1, 1],
            "cp": [1, 4],
            "trestbps": [145, 160],
            "chol": [233, 286],
            "fbs": [1, 0],
            "restecg": [2, 2],
            "thalach": [150, 108],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.5],
            "slope": [3, 2],
            "ca": [0, 3],
            "thal": [6, 3],
            "target": [0, 2],
        }
    )

    cleaned = clean_dataset(data)

    assert cleaned["target"].tolist() == [0, 1]


def test_clean_dataset_removes_missing_values():
    data = pd.DataFrame(
        {
            "age": [63, 67],
            "sex": [1, 1],
            "cp": [1, 4],
            "trestbps": [145, 160],
            "chol": [233, 286],
            "fbs": [1, 0],
            "restecg": [2, 2],
            "thalach": [150, 108],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.5],
            "slope": [3, 2],
            "ca": [None, 3],
            "thal": [6, 3],
            "target": [0, 2],
        }
    )

    cleaned = clean_dataset(data)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["age"] == 67
