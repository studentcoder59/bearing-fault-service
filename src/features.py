"""Windowing and feature extraction for vibration signals (12 kHz drive-end)."""
import numpy as np
from scipy.stats import kurtosis, skew

FS = 12_000
WINDOW = 2048
STRIDE = 1024
N_BANDS = 8  # equal-width FFT bands from 0 to FS/2

FEATURE_NAMES = (
    ["rms", "peak_to_peak", "crest_factor", "kurtosis", "skewness"]
    + [f"band_{i}" for i in range(N_BANDS)]
)


def make_windows(signal: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Split a 1-D signal into overlapping windows -> (n_windows, window)."""
    signal = np.asarray(signal, dtype=float).ravel()
    if len(signal) < window:
        return np.empty((0, window))
    n = 1 + (len(signal) - window) // stride
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return signal[idx]


def extract_features(windows: np.ndarray) -> np.ndarray:
    """(n_windows, window) -> (n_windows, len(FEATURE_NAMES))."""
    w = np.atleast_2d(windows)
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    p2p = w.max(axis=1) - w.min(axis=1)
    crest = np.max(np.abs(w), axis=1) / (rms + 1e-12)
    kurt = kurtosis(w, axis=1)
    skw = skew(w, axis=1)

    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) ** 2
    bands = np.array_split(np.arange(spec.shape[1]), N_BANDS)
    band_energy = np.stack([np.log10(spec[:, b].sum(axis=1) + 1e-12) for b in bands], axis=1)

    return np.column_stack([rms, p2p, crest, kurt, skw, band_energy])
