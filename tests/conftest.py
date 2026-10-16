"""Shared fixtures. Tests use synthetic data so they run without the real dataset."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor


def make_raw_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame in the *raw* UCI layout (label column `num`, '?'-style NaNs)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(246, 50, n).round(),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.normal(150, 22, n).round(),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.gamma(1.2, 0.9, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    score = (0.04 * (df["age"] - 54) + 0.8 * (df["cp"] == 4) + 0.9 * df["exang"]
             + 0.7 * df["oldpeak"] + 0.6 * df["ca"] - 0.02 * (df["thalach"] - 150))
    df["num"] = (score + rng.normal(0, 0.8, n) > 1.0).astype(int) * rng.integers(1, 5, n)
    df.loc[3, "ca"] = np.nan
    df.loc[7, "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_frame():
    return make_raw_frame()


@pytest.fixture(scope="session")
def clean_frame(raw_frame):
    from src.data import clean
    return clean(raw_frame)


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory, clean_frame):
    """Train a tiny model once and save it where the API can load it."""
    X = clean_frame[config.FEATURE_COLUMNS]
    y = clean_frame[config.TARGET]
    model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture()
def sample_patient():
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }
