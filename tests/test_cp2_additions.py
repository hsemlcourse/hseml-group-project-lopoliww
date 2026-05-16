import numpy as np
import pandas as pd

from src.modeling import QuantileClipper, build_preprocessor, feature_columns


def test_quantile_clipper_limits_train_numeric_outliers():
    x = pd.DataFrame({"a": [1, 2, 3, 4, 10_000], "b": [0, 0, 1, 1, 1]})
    clipper = QuantileClipper(lower_quantile=0.0, upper_quantile=0.8).fit(x)
    transformed = clipper.transform(pd.DataFrame({"a": [50_000], "b": [1]}))
    assert np.isclose(transformed.loc[0, "a"], x["a"].quantile(0.8))


def test_build_preprocessor_accepts_clipping_flag():
    x = pd.DataFrame({"num": [1.0, 2.0, 100.0], "cat": ["a", "b", "a"]})
    preprocessor = build_preprocessor(x, clip_outliers=True)
    transformed = preprocessor.fit_transform(x)
    assert transformed.shape[0] == 3


def test_feature_columns_keeps_engineered_features_only():
    df = pd.DataFrame(
        {
            "course_id": [1],
            "learners_count": [10],
            "target_log_learners_count": [2.4],
            "split": ["train"],
            "rating": [4.7],
            "step_type": ["text"],
            "title": ["raw text"],
        }
    )
    cols = feature_columns(df)
    assert "rating" in cols
    assert "step_type" in cols
    assert "title" not in cols
    assert "course_id" not in cols
