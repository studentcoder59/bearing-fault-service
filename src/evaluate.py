"""Harder evaluation: leave-one-load-out CV + noise robustness. Saves plots to results/."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay

from .data import build_dataset
from .features import extract_features

RESULTS = Path("results")


def add_noise(windows, snr_db, rng):
    """Add white Gaussian noise at a given signal-to-noise ratio (dB)."""
    power = np.mean(windows ** 2, axis=1, keepdims=True)
    noise_std = np.sqrt(power / (10 ** (snr_db / 10)))
    return windows + rng.normal(size=windows.shape) * noise_std


def fit(X, y):
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    meta, windows = build_dataset()
    y = meta["label"].to_numpy()
    load = meta["load_hp"].to_numpy()
    X = extract_features(windows)

    print("\n== Leave-one-load-out (train on 3 loads, test on the 4th) ==")
    for held in sorted(set(load)):
        te = load == held
        clf = fit(X[~te], y[~te])
        f1 = f1_score(y[te], clf.predict(X[te]), average="macro")
        print(f"held-out load {held} HP: macro-F1 = {f1:.3f}")

    print("\n== Noise robustness (train loads 0-2 clean, test load 3 with noise) ==")
    te = load == 3
    clf = fit(X[~te], y[~te])
    snrs, scores = [None, 20, 10, 5, 0, -5], []
    for snr in snrs:
        w = windows[te] if snr is None else add_noise(windows[te], snr, rng)
        f1 = f1_score(y[te], clf.predict(extract_features(w)), average="macro")
        scores.append(f1)
        print(f"SNR {'clean' if snr is None else str(snr) + ' dB':>6}: macro-F1 = {f1:.3f}")

    labels = ["clean" if s is None else f"{s} dB" for s in snrs]
    plt.figure(figsize=(6, 4))
    plt.plot(labels, scores, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Noise level (SNR)")
    plt.ylabel("Macro-F1 on unseen load")
    plt.title("RandomForest robustness to noise")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "noise_robustness.png", dpi=150)

    ConfusionMatrixDisplay.from_estimator(clf, X[te], y[te], xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    print("\nSaved plots to results/")


if __name__ == "__main__":
    main()