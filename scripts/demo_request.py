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

# Drift: recordings are mixed from all 4 states; each window is compared with training windows of its own class
def recording(files):
    return np.concatenate([load_signal(RAW_DIR / f)[:40_000] for f in files])


cases = [
    ("training load 2 HP (should be OK)", recording(["99.mat", "107.mat", "120.mat", "132.mat"])),
    ("unseen load 3 HP", recording(FILES.values())),
    ("load 3 HP amplified 3x (should drift)", recording(FILES.values()) * 3),
]
for name, sig in cases:
    r = httpx.post(f"{URL}/drift", json={"signal": sig.tolist()}, timeout=60).json()
    print(f"\n== /drift: {name} ==")
    print("drift_detected:", r["drift_detected"])
    for cname, c in r["classes"].items():
        print(f"   {cname:10} windows={c['n_windows']:3}  {c['status']}  drifted features: {c.get('n_features_drifted', '-')}")
