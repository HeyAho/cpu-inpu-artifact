#!/usr/bin/env python3
"""Feature extraction and 5-fold RF classification for Experiment 04."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


def one_file(path: Path, baseline: float, active: float) -> dict:
    df = pd.read_csv(path)
    t = df.Timestamp.to_numpy() - float(df.Timestamp.iloc[0])
    base = df[t < baseline]
    on = df[(t >= baseline) & (t <= baseline + active)]
    if len(on) == 0:
        raise ValueError(f"no active samples in {path}")
    a = on.ANE.to_numpy(float)
    b = base.ANE.to_numpy(float) if len(base) else np.array([0.0])
    f = {
        "ane_mean": a.mean(), "ane_median": np.median(a), "ane_std": a.std(),
        "ane_p10": np.percentile(a, 10), "ane_p90": np.percentile(a, 90),
        "ane_peak": a.max(), "ane_delta": a.mean() - b.mean(),
        "ane_iqr": np.percentile(a, 75) - np.percentile(a, 25),
    }
    if "GPU_Freq" in on:
        f["gpu_freq_mean"] = on["GPU_Freq"].mean()
        f["gpu_freq_max"] = on["GPU_Freq"].max()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--output", default="classification_results.json")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--active", type=float, default=15.0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    data = Path(args.data); data = data if data.is_absolute() else root / data
    rows, labels, files = [], [], []
    for label in ("black", "white"):
        paths = sorted(data.glob(f"resnet_power_{label}_r*.csv"), key=lambda p: int(re.search(r"r(\d+)", p.name).group(1)))
        for p in paths:
            rows.append(one_file(p, args.baseline, args.active)); labels.append(label); files.append(p.name)
    if len(set(labels)) < 2 or min(labels.count("black"), labels.count("white")) < 5:
        raise SystemExit(f"need at least 5 samples per class, found {dict((x, labels.count(x)) for x in set(labels))}")
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.array(labels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    # Predictions are generated fold-by-fold, never from a model trained on that row.
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        fold = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
        fold.fit(X.iloc[tr], y[tr]); pred[te] = fold.predict(X.iloc[te])
    out = {
        "experiment": "04_npu_power_black_white",
        "platform": "M5 Pro",
        "n_black": int(np.sum(y == "black")), "n_white": int(np.sum(y == "white")),
        "features": list(X.columns), "feature_matrix_rows": len(X),
        "cv_5fold_scores": [float(s) for s in scores],
        "cv_5fold_mean": float(scores.mean()), "cv_5fold_std": float(scores.std()),
        "confusion_matrix_labels": ["black", "white"],
        "confusion_matrix": confusion_matrix(y, pred, labels=["black", "white"]).tolist(),
        "oof_accuracy": float(accuracy_score(y, pred)),
        "black_ane_mean": float(X.loc[y == "black", "ane_mean"].mean()),
        "black_ane_std": float(X.loc[y == "black", "ane_mean"].std(ddof=0)),
        "white_ane_mean": float(X.loc[y == "white", "ane_mean"].mean()),
        "white_ane_std": float(X.loc[y == "white", "ane_mean"].std(ddof=0)),
        "feature_importance": {k: float(v) for k, v in zip(X.columns, clf.feature_importances_)},
        "files": files,
    }
    out_path = Path(args.output); out_path = out_path if out_path.is_absolute() else root / out_path
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
