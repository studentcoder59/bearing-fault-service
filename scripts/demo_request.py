"""Send real CWRU windows to the running API. Start the server first:
    py -3.14 -m uvicorn app.main:app
Then run:  py -3.14 -m scripts.demo_request
"""
import httpx
import numpy as np

from src.data import RAW_DIR, load_signal
from src.features import WINDOW

URL = "http://127.0.0.1:8000"
# One load-3 HP recording per class (the load the models never trained on)
FILES = {"normal": "100.mat", "inner_race": "108.mat", "ball": "121.mat", "outer_race": "133.mat"}

signals = {label: load_signal(RAW_DIR / name) for label, name in FILES.items()}

print("== /health ==")
print(httpx.get(f"{URL}/health").json())

print("\n== /predict ==")
for model in ("rf", "cnn"):
    for true_label, sig in signals.items():
        window = sig[5000:5000 + WINDOW].tolist()
        r = httpx.post(f"{URL}/predict", json={"signal": window, "model": model}, timeout=30)
        if r.status_code != 200:
            print(f"{model:3} true={true_label:10} -> HTTP {r.status_code}: {r.json()}")
            continue
        b = r.json()
        print(f"{model:3} true={true_label:10} -> {b['label']:10} conf={b['confidence']:.3f}  {b['latency_ms']:.2f} ms")

mixed = np.concatenate([s[:30_000] for s in signals.values()])
for name, sig in [("recording mixing all 4 states", mixed), ("same recording amplified 3x", mixed * 3)]:
    r = httpx.post(f"{URL}/drift", json={"signal": sig.tolist()}, timeout=60).json()
    print(f"\n== /drift: {name} ==")
    print("drift_detected:", r["drift_detected"], "| features drifted:", r["n_features_drifted"], "of", len(r["features"]))
