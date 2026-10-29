---
title: Bearing Fault Detection
emoji: ⚙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Bearing Fault Detection Service

End-to-end ML service: vibration signals -> features -> model -> REST API -> Docker -> deployed, with drift monitoring.

## Setup
```
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Data
Download the 16 drive-end 12 kHz files listed in `data/manifest.csv` from the
CWRU Bearing Data Center into `data/raw/` (files `97.mat`, `98.mat`, ...).
Verify the file numbers against the table on the CWRU website.

## Run
```
python -m src.train_baseline
pytest
```

## Status
- [x] Day 1: features + baseline (leakage-safe split by motor load)
- [ ] Day 2: PyTorch 1D-CNN + comparison
- [ ] Day 3: FastAPI + drift + tests + Docker
- [ ] Day 4: Deploy + results table


## API
Start: `uvicorn app.main:app` · Interactive docs: `/docs`

| Endpoint | Purpose |
|---|---|
| `GET /health` | Which models are loaded |
| `POST /predict` | Body: `{"signal": [2048 floats], "model": "cnn" or "rf"}`. Returns label, confidence, latency |
| `POST /drift` | Body: `{"signal": [at least 31,744 floats]}`. Returns a per-class drift report |

## Drift monitoring (experimental)
Each window is classified, then compared with training windows of the same class.
A feature counts as drifted if more than 25% of windows fall outside the training 1-99 percentile range.
Drift is reported when at least 3 features drift. A PSI/KS version false-alarmed on real data
(one recording is much narrower than the pooled training data), so it was replaced.

Observed: unseen load 3 HP flagged drift for inner_race while classification accuracy stayed 100%.

Limits: thresholds are heuristics set on CWRU only; classes with fewer than 30 windows are not checked.