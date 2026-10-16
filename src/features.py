"""Feature engineering and the scikit-learn preprocessing transformer."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features.

    hr_ratio: peak heart rate achieved / age-predicted max (220 - age). Values
    well below 1 suggest poor chronotropic response, a known cardiac risk marker.
    Defined at module level so the fitted pipeline can be pickled and reloaded.
    """
    out = df.copy()
    predicted_max = (220.0 - out["age"]).clip(lower=1.0)
    out["hr_ratio"] = out["thalach"] / predicted_max
    return out


def build_preprocessor() -> Pipeline:
    """Feature creation + imputation + scaling + one-hot encoding."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("add_features", FunctionTransformer(add_features, validate=False)),
        ("columns", column_transformer),
    ])


def get_feature_names(fitted_preprocessor: Pipeline) -> np.ndarray:
    """Output column names after fitting (used for importance plots)."""
    names = fitted_preprocessor.named_steps["columns"].get_feature_names_out()
    return np.array([n.split("__", 1)[-1] for n in names])
