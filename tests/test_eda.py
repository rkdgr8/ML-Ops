import pandas as pd

from src.eda import summarize_dataset


def test_summarize_dataset_returns_expected_counts():
    data = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "chol": [200, 220, 240],
            "target": [0, 1, 1],
        }
    )

    summary = summarize_dataset(data)

    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert summary["missing_values"] == 0
    assert summary["target_distribution"] == {0: 1, 1: 2}
