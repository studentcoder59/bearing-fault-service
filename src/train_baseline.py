"""Day 1: RandomForest baseline with a leakage-safe split (train loads 0-2, test load 3)."""
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .data import build_dataset
from .features import extract_features, FEATURE_NAMES

TEST_LOAD = 3


def main():
    meta, windows = build_dataset()
    X = extract_features(windows)
    y = meta["label"].to_numpy()
    test = (meta["load_hp"] == TEST_LOAD).to_numpy()

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X[~test], y[~test])
    pred = clf.predict(X[test])

    print(f"train windows: {(~test).sum()}  test windows: {test.sum()}")
    print(classification_report(y[test], pred, digits=3))
    print(confusion_matrix(y[test], pred, labels=clf.classes_))

    joblib.dump(clf, "models/rf.joblib")
    # Training feature values, reused later by the /drift endpoint
    ref = {n: X[~test][:, i].tolist() for i, n in enumerate(FEATURE_NAMES)}
    with open("models/reference_features.json", "w") as f:
        json.dump(ref, f)


if __name__ == "__main__":
    main()
