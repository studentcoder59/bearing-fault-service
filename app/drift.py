"""Data-drift check: are incoming feature values still inside the range seen in training?

For each feature we measure the share of incoming windows that fall outside the
1st-99th percentile range of the training windows of the same class. Healthy data
should have only a few percent outside; a real shift pushes most windows outside.
(An earlier PSI/KS version false-alarmed on real data: one recording is much narrower
than the pooled training data, which those tests read as 'drift'.)
"""
import numpy as np

from src.features import FEATURE_NAMES

RANGE_PERCENTILES = (1, 99)
OUT_OF_RANGE_THRESHOLD = 0.25  # feature drifted if >25% of windows fall outside the training range
MIN_FEATURES_DRIFTED = 3       # report drift only if several features shift together


def out_of_range_rate(reference: np.ndarray, current: np.ndarray) -> float:
    lo, hi = np.percentile(reference, RANGE_PERCENTILES)
    return float(np.mean((current < lo) | (current > hi)))


def drift_report(reference: dict, features: np.ndarray) -> dict:
    per_feature = {}
    for i, name in enumerate(FEATURE_NAMES):
        rate = out_of_range_rate(np.asarray(reference[name], dtype=float), features[:, i])
        per_feature[name] = {"out_of_range": round(rate, 3), "drifted": bool(rate > OUT_OF_RANGE_THRESHOLD)}
    n_drifted = sum(v["drifted"] for v in per_feature.values())
    return {
        "n_windows": int(len(features)),
        "n_features_drifted": n_drifted,
        "drift_detected": n_drifted >= MIN_FEATURES_DRIFTED,
        "features": per_feature,
    }
