"""Inference helpers shared by the notebook, CLI and API."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src import config


def load_model(path=None):
    path = Path(path or os.environ.get("MODEL_PATH", config.MODEL_PATH))
    return joblib.load(path)


def load_metadata(path=None) -> dict:
    path = Path(path or os.environ.get("METADATA_PATH", config.METADATA_PATH))
    return json.loads(path.read_text()) if path.exists() else {}


def predict_records(model, records: list[dict]) -> list[dict]:
    """Score a list of patient dicts. Returns label, confidence and P(disease)."""
    frame = pd.DataFrame(records)[config.FEATURE_COLUMNS]
    p_disease = model.predict_proba(frame)[:, 1]
    results = []
    for p in p_disease:
        label = int(p >= 0.5)
        results.append({
            "prediction": label,
            "label": "heart disease likely" if label else "no heart disease likely",
            "confidence": round(float(p if label else 1 - p), 4),
            "probability_disease": round(float(p), 4),
        })
    return results


if __name__ == "__main__":
    sample = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }
    print(predict_records(load_model(), [sample])[0])
