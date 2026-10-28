"""Load CWRU .mat files listed in data/manifest.csv into windows + labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .features import make_windows

RAW_DIR = Path("data/raw")
MANIFEST = Path("data/manifest.csv")


def load_signal(path: Path) -> np.ndarray:
    """Return the drive-end accelerometer signal from a CWRU .mat file."""
    mat = loadmat(path)
    keys = [k for k in mat if k.endswith("DE_time")]
    if not keys:
        raise KeyError(f"No *_DE_time key in {path.name}. Keys: {list(mat)}")
    return mat[keys[0]].ravel()


def build_dataset(raw_dir: Path = RAW_DIR, manifest: Path = MANIFEST):
    """Return (meta DataFrame with label/load_hp/file per window, windows array)."""
    rows, wins = [], []
    for rec in pd.read_csv(manifest).itertuples():
        w = make_windows(load_signal(raw_dir / rec.file))
        wins.append(w)
        rows += [(rec.label, rec.load_hp, rec.file)] * len(w)
    meta = pd.DataFrame(rows, columns=["label", "load_hp", "file"])
    return meta, np.vstack(wins)
