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
