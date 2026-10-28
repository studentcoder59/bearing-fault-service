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
    # reference pooled over 3 "motor loads" (scale x0.9/1.0/1.1), normal class 3x longer like real data
    for idx, scale in enumerate(SCALES):
        for mult in (0.9, 1.0, 1.1):
            n = 130_000 if CLASSES[idx] == "normal" else 40_000
            f = extract_features(make_windows(_signal(rng, scale * mult, n)))
            X.append(f)
            y += [idx] * len(f)
    X = np.vstack(X)
    y = np.array(y)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    joblib.dump(rf, d / "rf.joblib")
    json.dump({"classes": CLASSES, "std": 0.25}, open(d / "cnn_meta.json", "w"))
    ref = {CLASSES[k]: {n: X[y == k][:, i].tolist() for i, n in enumerate(FEATURE_NAMES)} for k in range(4)}
    json.dump(ref, open(d / "reference_by_class.json", "w"))

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


def _recording(seed, mult, scales=SCALES, n=40_000):
    rng = np.random.default_rng(seed)
    return np.concatenate([_signal(rng, sc * mult, n) for sc in scales])


@pytest.mark.parametrize("mult", [0.9, 1.0, 1.1])
def test_no_false_alarm_for_single_training_load(client, mult):
    """One recording is narrower than the pooled training data; that must not count as drift."""
    body = client.post("/drift", json={"signal": _recording(3, mult).tolist()}).json()
    assert body["drift_detected"] is False
    assert all(c["status"] == "ok" for c in body["classes"].values())


def test_single_state_recording_ok(client):
    sig = _signal(np.random.default_rng(5), 0.1, 50_000)  # healthy only
    body = client.post("/drift", json={"signal": sig.tolist()}).json()
    assert body["drift_detected"] is False
    assert list(body["classes"]) == ["normal"]


@pytest.mark.parametrize("mult", [1.3, 3.0])
def test_drift_detected_outside_training_range(client, mult):
    body = client.post("/drift", json={"signal": _recording(4, mult).tolist()}).json()
    assert body["drift_detected"] is True


def test_drift_too_short_rejected(client):
    assert client.post("/drift", json={"signal": [0.0] * 5000}).status_code == 422
