import numpy as np
import pandas as pd

from src import config
from src.data import clean, load_raw, split_xy


def test_clean_creates_binary_target(raw_frame):
    out = clean(raw_frame)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns


def test_clean_maps_severity_levels_to_one(raw_frame):
    out = clean(raw_frame.assign(num=[0, 1, 2, 3, 4] * 24))
    # every non-zero severity collapses to 1
    assert out[config.TARGET].isin([0, 1]).all()
    assert out[config.TARGET].sum() > 0


def test_clean_drops_duplicates(raw_frame):
    doubled = pd.concat([raw_frame, raw_frame.iloc[:10]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_frame))


def test_clean_keeps_missing_for_pipeline_imputation(raw_frame):
    out = clean(raw_frame)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1


def test_split_xy_returns_expected_columns(clean_frame):
    X, y = split_xy(clean_frame)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert len(X) == len(y)


def test_load_raw_parses_question_marks(tmp_path):
    row = "63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n"
    path = tmp_path / "raw.data"
    path.write_text(row)
    df = load_raw(path)
    assert list(df.columns) == config.RAW_COLUMNS
    assert np.isnan(df.loc[0, "ca"])