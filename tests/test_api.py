import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_model_path, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_path))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "missing.json"))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert "request_id" in body
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    del sample_patient["ca"], sample_patient["thal"]
    assert client.post("/predict", json=sample_patient).status_code == 200


def test_predict_rejects_missing_required_field(client, sample_patient):
    del sample_patient["age"]
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_out_of_range_value(client, sample_patient):
    sample_patient["age"] = -3
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_metrics_endpoint_exposes_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_request_duration_seconds_bucket" in text
