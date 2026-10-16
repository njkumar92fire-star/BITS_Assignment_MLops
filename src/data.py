"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
import logging
from pathlib import Path

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)


def download_raw(dest: Path = config.RAW_PATH, url: str = config.UCI_URL) -> Path:
    """Download the processed Cleveland file from the UCI repository."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved raw data to %s", dest)
    return dest


def load_raw(path: Path = config.RAW_PATH) -> pd.DataFrame:
    """Read the raw file; '?' marks missing values in the UCI files."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame with a binary `target` column.

    - original label `num` (0-4) is collapsed: 0 = no disease, 1-4 = disease
    - duplicates are dropped
    - missing values are left in place on purpose: they are imputed inside
      the sklearn Pipeline so that no information leaks from the test split
    """
    out = df.copy()
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    for col in config.CATEGORICAL_FEATURES + config.BINARY_FEATURES:
        out[col] = out[col].astype("float64")
    return out


def prepare_clean_dataset(
    raw_path: Path = config.RAW_PATH, out_path: Path = config.CLEAN_PATH
) -> pd.DataFrame:
    """Raw file -> cleaned CSV on disk."""
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return df


def load_clean(path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[config.FEATURE_COLUMNS], df[config.TARGET]
