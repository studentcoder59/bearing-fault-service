"""Data-drift check: compare incoming feature distributions with the training distribution."""
import numpy as np
from scipy.stats import ks_2samp

from src.features import FEATURE_NAMES

PSI_THRESHOLD = 0.2      # common rule of thumb: >0.2 = significant shift
MIN_FEATURES_DRIFTED = 3  # flag overall drift only if several features shift together
BINS = 5                 # few bins: PSI is biased upward on small batches


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    """Population Stability Index between two 1-D samples (bins from reference quantiles)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, 1e-4, None), np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(reference: dict, features: np.ndarray) -> dict:
    per_feature = {}
    for i, name in enumerate(FEATURE_NAMES):
        ref = np.asarray(reference[name], dtype=float)
        cur = features[:, i]
        stat, p = ks_2samp(ref, cur)
        score = psi(ref, cur)
        per_feature[name] = {
            "psi": round(score, 4),
            "ks_statistic": round(float(stat), 4),
            "ks_p_value": round(float(p), 6),
            "drifted": bool(score > PSI_THRESHOLD),
        }
    n_drifted = sum(v["drifted"] for v in per_feature.values())
    return {
        "n_windows": int(len(features)),
        "n_features_drifted": n_drifted,
        "drift_detected": n_drifted >= MIN_FEATURES_DRIFTED,
        "features": per_feature,
    }
