import pandas as pd

from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    split_features_target,
)


def test_split_features_target_separates_target_column():
    data = pd.DataFrame(
        {
            "age": [50, 60],
            "chol": [200, 220],
            "target": [0, 1],
        }
    )

    features, target = split_features_target(data)

    assert "target" not in features.columns
    assert target.tolist() == [0, 1]


def test_build_preprocessor_transforms_expected_columns():
    data = pd.DataFrame(
        {
            **{column: [1, 2] for column in NUMERIC_FEATURES},
            **{column: [0, 1] for column in CATEGORICAL_FEATURES},
        }
    )

    transformed = build_preprocessor().fit_transform(data)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
