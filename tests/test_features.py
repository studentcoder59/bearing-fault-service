import numpy as np
from src.features import make_windows, extract_features, WINDOW, FEATURE_NAMES


def test_window_shape():
    w = make_windows(np.random.randn(10_000))
    assert w.shape[1] == WINDOW


def test_feature_shape_and_finite():
    f = extract_features(make_windows(np.random.randn(10_000)))
    assert f.shape[1] == len(FEATURE_NAMES)
    assert np.isfinite(f).all()


def test_short_signal_gives_no_windows():
    assert make_windows(np.random.randn(100)).shape[0] == 0
