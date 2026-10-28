"""Day 2: 1D-CNN vs RandomForest on the same leakage-safe split.
Compares accuracy, noise robustness, latency and model size."""
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from .data import build_dataset
from .evaluate import add_noise
from .features import extract_features
from .model import CNN1D

RESULTS = Path("results")
TEST_LOAD = 3
EPOCHS = 15
BATCH = 64
SNRS = [None, 20, 10, 5, 0, -5]

torch.manual_seed(42)
np.random.seed(42)


def to_tensor(w, std):
    return torch.tensor(w / std, dtype=torch.float32).unsqueeze(1)


def predict_cnn(model, w, std):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(w, std)).argmax(1).numpy()


def main():
    RESULTS.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    meta, windows = build_dataset()
    classes, y = np.unique(meta["label"].to_numpy(), return_inverse=True)
    te = (meta["load_hp"] == TEST_LOAD).to_numpy()
    Wtr, Wte, ytr, yte = windows[~te], windows[te], y[~te], y[te]
    std = float(Wtr.std())

    # ---- RandomForest (same as baseline) ----
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(extract_features(Wtr), ytr)
    rf.n_jobs = 1  # fair single-request latency
    joblib.dump(rf, "models/rf.joblib")

    # ---- CNN ----
    model = CNN1D(len(classes))
    counts = np.bincount(ytr)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)  # handles imbalance
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, Ytr = to_tensor(Wtr, std), torch.tensor(ytr)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch:2d}/{EPOCHS}  train loss {total / len(Xtr):.4f}")

    torch.save(model.state_dict(), "models/cnn.pt")
    with open("models/cnn_meta.json", "w") as f:
        json.dump({"std": std, "classes": classes.tolist()}, f)

    print("\n== CNN on unseen load 3 ==")
    print(classification_report(yte, predict_cnn(model, Wte, std), target_names=classes, digits=3))

    # ---- Noise robustness, both models ----
    rng = np.random.default_rng(0)
    rf_scores, cnn_scores = [], []
    for snr in SNRS:
        w = Wte if snr is None else add_noise(Wte, snr, rng)
        rf_scores.append(f1_score(yte, rf.predict(extract_features(w)), average="macro"))
        cnn_scores.append(f1_score(yte, predict_cnn(model, w, std), average="macro"))
    print("== Noise robustness (macro-F1) ==")
    for snr, a, b in zip(SNRS, rf_scores, cnn_scores):
        print(f"SNR {'clean' if snr is None else str(snr) + ' dB':>6}:  RF {a:.3f}   CNN {b:.3f}")

    # ---- Latency (1 window per request) and size ----
    one = Wte[:1]
    n = 200
    t = time.perf_counter()
    for _ in range(n):
        rf.predict(extract_features(one))
    rf_ms = (time.perf_counter() - t) / n * 1000
    t = time.perf_counter()
    for _ in range(n):
        predict_cnn(model, one, std)
    cnn_ms = (time.perf_counter() - t) / n * 1000
    rf_kb = os.path.getsize("models/rf.joblib") / 1024
    cnn_kb = os.path.getsize("models/cnn.pt") / 1024
    n_params = sum(p.numel() for p in model.parameters())

    print("\n== Summary ==")
    print(f"{'':14}{'macro-F1':>10}{'latency ms':>12}{'size KB':>10}")
    print(f"{'RandomForest':14}{rf_scores[0]:>10.3f}{rf_ms:>12.2f}{rf_kb:>10.0f}")
    print(f"{'1D-CNN':14}{cnn_scores[0]:>10.3f}{cnn_ms:>12.2f}{cnn_kb:>10.0f}   ({n_params:,} parameters)")

    labels = ["clean" if s is None else f"{s} dB" for s in SNRS]
    plt.figure(figsize=(6, 4))
    plt.plot(labels, rf_scores, marker="o", label="RandomForest")
    plt.plot(labels, cnn_scores, marker="s", label="1D-CNN")
    plt.ylim(0, 1.05)
    plt.xlabel("Noise level (SNR)")
    plt.ylabel("Macro-F1 on unseen load")
    plt.title("Noise robustness: RandomForest vs 1D-CNN")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "noise_comparison.png", dpi=150)
    print("\nSaved results/noise_comparison.png")


if __name__ == "__main__":
    main()
