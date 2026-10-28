"""Build the per-class reference distributions used by /drift (training loads only).
Run once after training:  py -3.14 -m scripts.make_reference
"""
import json

import numpy as np

from src.data import build_dataset
from src.features import FEATURE_NAMES, extract_features

TEST_LOAD = 3  # same split as training: loads 0-2 are the reference

meta, windows = build_dataset()
train = (meta["load_hp"] != TEST_LOAD).to_numpy()
X = extract_features(windows[train])
y = meta["label"].to_numpy()[train]

reference = {
    c: {n: X[y == c][:, i].round(5).tolist() for i, n in enumerate(FEATURE_NAMES)}
    for c in np.unique(y)
}
with open("models/reference_by_class.json", "w") as f:
    json.dump(reference, f)
print({c: int((y == c).sum()) for c in np.unique(y)}, "-> models/reference_by_class.json")
