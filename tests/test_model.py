import joblib
import numpy as np

from src import config
from src.predict import predict_records
from src.train import candidate_models, make_pipeline


def test_every_candidate_trains_and_predicts_probabilities(clean_frame):
    X = clean_frame[config.FEATURE_COLUMNS]
    y = clean_frame[config.TARGET]
    for name, (estimator, grid) in candidate_models().items():
        assert grid, f"{name} has an empty grid"
        model = make_pipeline(estimator).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_saved_pipeline_round_trips(trained_model_path, clean_frame):
    X = clean_frame[config.FEATURE_COLUMNS]
    model = joblib.load(trained_model_path)
    again = joblib.load(trained_model_path)
    assert np.allclose(model.predict_proba(X), again.predict_proba(X))


def test_predict_records_schema(trained_model_path, sample_patient):
    model = joblib.load(trained_model_path)
    out = predict_records(model, [sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert 0.0 <= out["probability_disease"] <= 1.0


def test_predict_handles_missing_optional_fields(trained_model_path, sample_patient):
    model = joblib.load(trained_model_path)
    sample_patient.update({"ca": None, "thal": None})
    out = predict_records(model, [sample_patient])[0]
    assert out["prediction"] in (0, 1)
