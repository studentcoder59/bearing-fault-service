import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURE_NAMES, WINDOW, extract_features, make_windows

CLASSES = ["ball", "inner_race", "normal", "outer_race"]
SCALES = [0.2, 0.3, 0.1, 0.4]


def _signal(rng, scale, n):
    return rng.normal(0, scale, n)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Tiny synthetic models so tests don't need the real dataset."""
    d = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    X, y = [], []
    for idx, scale in enumerate(SCALES):
        f = extract_features(make_windows(_signal(rng, scale, 80_000)))
        X.append(f)
        y += [idx] * len(f)
    X = np.vstack(X)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    joblib.dump(rf, d / "rf.joblib")
    json.dump({"classes": CLASSES, "std": 0.25}, open(d / "cnn_meta.json", "w"))
    ref = {n: X[:, i].tolist() for i, n in enumerate(FEATURE_NAMES)}
    json.dump(ref, open(d / "reference_features.json", "w"))

    import os
    os.environ["MODEL_DIR"] = str(d)
    os.environ["ENABLE_CNN"] = "0"
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models"] == {"rf": True, "cnn": False}


def test_predict_returns_valid_label(client):
    sig = _signal(np.random.default_rng(1), 0.4, WINDOW).tolist()
    r = client.post("/predict", json={"signal": sig})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] in CLASSES
    assert 0 <= body["confidence"] <= 1
    assert body["model"] == "rf"  # CNN not loaded -> falls back


def test_predict_wrong_length_rejected(client):
    assert client.post("/predict", json={"signal": [0.1] * 100}).status_code == 422


def test_cnn_requested_but_not_loaded(client):
    sig = _signal(np.random.default_rng(2), 0.1, WINDOW).tolist()
    assert client.post("/predict", json={"signal": sig, "model": "cnn"}).status_code == 503


def test_drift_not_detected_on_similar_data(client):
    rng = np.random.default_rng(3)
    sig = np.concatenate([_signal(rng, s, 30_000) for s in SCALES])
    r = client.post("/drift", json={"signal": sig.tolist()})
    assert r.status_code == 200
    assert r.json()["drift_detected"] is False


def test_drift_detected_on_shifted_data(client):
    rng = np.random.default_rng(4)
    sig = np.concatenate([_signal(rng, s * 5, 30_000) for s in SCALES])
    r = client.post("/drift", json={"signal": sig.tolist()})
    assert r.json()["drift_detected"] is True


def test_drift_too_short_rejected(client):
    assert client.post("/drift", json={"signal": [0.0] * 5000}).status_code == 422
