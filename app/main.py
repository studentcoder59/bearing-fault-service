"""FastAPI service: bearing fault prediction (RandomForest + optional 1D-CNN) and drift monitoring."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.drift import drift_report
from src.features import STRIDE, WINDOW, extract_features, make_windows

MIN_DRIFT_WINDOWS = 30  # per predicted class; PSI is unreliable on tiny batches
MIN_DRIFT_SAMPLES = WINDOW + STRIDE * (MIN_DRIFT_WINDOWS - 1)

state: dict = {}


def load_models() -> None:
    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    meta = json.loads((model_dir / "cnn_meta.json").read_text())
    state["classes"] = meta["classes"]
    state["std"] = float(meta["std"])
    state["rf"] = joblib.load(model_dir / "rf.joblib")
    state["reference"] = json.loads((model_dir / "reference_by_class.json").read_text())
    state["cnn"] = None
    if os.environ.get("ENABLE_CNN", "1") != "0":
        try:
            import torch
            from src.model import CNN1D

            net = CNN1D(len(state["classes"]))
            net.load_state_dict(torch.load(model_dir / "cnn.pt", map_location="cpu"))
            net.eval()
            state["cnn"] = net
        except Exception as exc:  # e.g. torch not installed or weights missing
            print(f"WARNING: CNN not loaded, falling back to RandomForest only ({exc})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    state.clear()


app = FastAPI(
    title="Bearing Fault Detection API",
    description="Classifies motor bearing faults from a 2048-sample vibration window (12 kHz).",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=WINDOW, max_length=WINDOW,
                                description=f"Exactly {WINDOW} vibration samples")
    model: Optional[Literal["cnn", "rf"]] = Field(None, description="Defaults to cnn if loaded, else rf")


class DriftRequest(BaseModel):
    signal: list[float] = Field(..., min_length=MIN_DRIFT_SAMPLES, max_length=2_000_000,
                                description=f"A recording of at least {MIN_DRIFT_SAMPLES} samples ({MIN_DRIFT_WINDOWS} windows)")


def _class_name(c) -> str:
    return state["classes"][int(c)] if isinstance(c, (int, np.integer)) else str(c)


def _check_finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise HTTPException(status_code=422, detail="signal contains NaN or infinite values")
    return arr


def _predict_rf(x: np.ndarray):
    rf = state["rf"]
    proba = rf.predict_proba(extract_features(x[None, :]))[0]
    return [_class_name(c) for c in rf.classes_], proba


def _predict_cnn(x: np.ndarray):
    import torch

    t = torch.tensor(x / state["std"], dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        proba = torch.softmax(state["cnn"](t), dim=1)[0].numpy()
    return state["classes"], proba


@app.get("/health")
def health():
    return {"status": "ok", "models": {"rf": "rf" in state, "cnn": state.get("cnn") is not None}}


@app.post("/predict")
def predict(req: PredictRequest):
    x = _check_finite(req.signal)
    chosen = req.model or os.environ.get("DEFAULT_MODEL") or ("cnn" if state.get("cnn") is not None else "rf")
    if chosen == "cnn" and state.get("cnn") is None:
        raise HTTPException(status_code=503, detail="CNN model is not loaded on this server; use model='rf'")
    t0 = time.perf_counter()
    names, proba = _predict_cnn(x) if chosen == "cnn" else _predict_rf(x)
    latency_ms = (time.perf_counter() - t0) * 1000
    best = int(np.argmax(proba))
    return {
        "label": names[best],
        "confidence": round(float(proba[best]), 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(names, proba)},
        "model": chosen,
        "latency_ms": round(latency_ms, 3),
    }


@app.post("/drift")
def drift(req: DriftRequest):
    """Class-conditional drift: each window is classified, then windows of each predicted class
    are compared with the training windows of that same class (so the class mix cannot cause false alarms)."""
    x = _check_finite(req.signal)
    feats = extract_features(make_windows(x))
    names = np.array([_class_name(c) for c in state["rf"].predict(feats)])

    classes = {}
    for cname in sorted(set(names)):
        mask = names == cname
        n = int(mask.sum())
        if n < MIN_DRIFT_WINDOWS or cname not in state["reference"]:
            classes[cname] = {"n_windows": n, "status": f"insufficient data (need >= {MIN_DRIFT_WINDOWS} windows)"}
            continue
        rep = drift_report(state["reference"][cname], feats[mask])
        classes[cname] = {
            "n_windows": n,
            "status": "drift" if rep["drift_detected"] else "ok",
            "n_features_drifted": rep["n_features_drifted"],
            "drifted_features": [{"feature": k, "out_of_range": v["out_of_range"]} for k, v in rep["features"].items() if v["drifted"]],
        }
    checked = [c for c in classes.values() if c["status"] in ("ok", "drift")]
    return {
        "n_windows": int(len(feats)),
        "drift_detected": any(c["status"] == "drift" for c in checked) if checked else None,
        "classes": classes,
    }
