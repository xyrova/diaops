"""API tests against a tiny dummy model, so they run in CI without the real one."""

import json

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from src.features import EXCLUDED_FEATURES, FEATURE_RANGES, FEATURES

VALID = {name: FEATURE_RANGES[name][0] for name in FEATURES} | {"BMI": 25, "Age": 7}


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    tmp = tmp_path_factory.mktemp("model")

    # 5-tree model on random data: only the API plumbing is under test here.
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, len(FEATURES)))
    y = rng.integers(0, 2, size=200)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    model.get_booster().feature_names = FEATURES
    model.save_model(tmp / "model.json")

    (tmp / "params.yaml").write_text("evaluate:\n  threshold: 0.5\n")
    (tmp / "metrics.json").write_text(json.dumps({"recall": 0.5}))
    monkeypatch_module.setenv("MODEL_PATH", str(tmp / "model.json"))
    monkeypatch_module.setenv("PARAMS_PATH", str(tmp / "params.yaml"))
    monkeypatch_module.setenv("METRICS_PATH", str(tmp / "metrics.json"))
    monkeypatch_module.setenv("MODEL_VERSION", "test")

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "test"
    assert body["features"] == FEATURES
    assert body["test_metrics"] == {"recall": 0.5}
    assert len(body["model_md5"]) == 32


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_level"] in {"low", "elevated"}
    assert "not a diagnosis" in body["disclaimer"]


@pytest.mark.parametrize("field, value", [("Age", 14), ("GenHlth", 0), ("HighBP", 2)])
def test_predict_out_of_range(client, field, value):
    assert client.post("/predict", json=VALID | {field: value}).status_code == 422


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID.items() if k != "BMI"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unknown_field(client):
    assert client.post("/predict", json=VALID | {"Glucose": 120}).status_code == 422


def test_excluded_features_rejected(client):
    # Income etc. are in the raw data but deliberately not model inputs.
    for name in EXCLUDED_FEATURES:
        assert client.post("/predict", json=VALID | {name: 1}).status_code == 422


def test_cors_header(client):
    r = client.get("/health", headers={"Origin": "http://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"
