import numpy as np
import pandas as pd

from src import config
from src.features import add_features, build_preprocessor, get_feature_names


def test_add_features_hr_ratio():
    df = pd.DataFrame({"age": [60.0], "thalach": [112.0]})
    out = add_features(df)
    assert np.isclose(out.loc[0, "hr_ratio"], 112 / 160)


def test_add_features_does_not_mutate_input():
    df = pd.DataFrame({"age": [60.0], "thalach": [112.0]})
    add_features(df)
    assert "hr_ratio" not in df.columns


def test_preprocessor_output_has_no_nans(clean_frame):
    X = clean_frame[config.FEATURE_COLUMNS]
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_preprocessor_scales_numeric_columns(clean_frame):
    X = clean_frame[config.FEATURE_COLUMNS]
    prep = build_preprocessor().fit(X)
    out = np.asarray(prep.transform(X), dtype=float)
    names = list(get_feature_names(prep))
    idx = names.index("age")
    assert abs(out[:, idx].mean()) < 1e-6
    assert abs(out[:, idx].std() - 1) < 1e-2


def test_preprocessor_ignores_unseen_category(clean_frame):
    X = clean_frame[config.FEATURE_COLUMNS]
    prep = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["cp"] = 99.0
    assert prep.transform(unseen).shape[0] == 1
