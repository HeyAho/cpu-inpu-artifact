#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PLATFORMS = ["intel", "amd"]
BASE_METRICS = [
    "npu_power_w",
    "npu_freq_mhz",
    "npu_activity_pct",
    "npu_work_per_s",
    "npu_irq_per_s",
    "npu_reads_mb_s",
    "npu_writes_mb_s",
]
CLASSIFIER_METRICS = [
    "npu_power_w",
    "npu_freq_mhz",
    "npu_activity_pct",
    "npu_work_per_s",
    "npu_reads_mb_s",
    "npu_writes_mb_s",
]


def load_trials(data_dir):
    rows = []
    for label in ["black", "white"]:
        paths = sorted(
            data_dir.glob(f"resnet_power_{label}_r*.csv"),
            key=lambda path: int(path.stem.rsplit("r", 1)[1]),
        )
        for path in paths:
            frame = pd.read_csv(path)
            infer = frame[frame["phase"] == "infer"]
            if infer.empty:
                continue
            row = {
                "label": label,
                "target": int(label == "white"),
                "rep": int(path.stem.rsplit("r", 1)[1]),
                "trial_start": frame["timestamp"].min(),
            }
            for metric in BASE_METRICS:
                if metric in infer:
                    row[metric] = infer[metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def best_threshold_accuracy(black, white):
    values = np.sort(np.unique(np.concatenate([black, white])))
    thresholds = np.concatenate((
        [values[0] - 1e-9],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + 1e-9],
    ))
    best = (0.0, np.nan, "")
    for threshold in thresholds:
        for direction in ["white_high", "white_low"]:
            if direction == "white_high":
                correct = np.count_nonzero(black < threshold) + np.count_nonzero(white >= threshold)
            else:
                correct = np.count_nonzero(black >= threshold) + np.count_nonzero(white < threshold)
            accuracy = correct / (len(black) + len(white))
            if accuracy > best[0]:
                best = (accuracy, threshold, direction)
    return best


def analyze_platform(platform, frame):
    threshold_rows = []
    available_metrics = [metric for metric in BASE_METRICS if metric in frame and frame[metric].notna().all()]
    for metric in available_metrics:
        black = frame[frame["target"] == 0][metric].to_numpy()
        white = frame[frame["target"] == 1][metric].to_numpy()
        accuracy, threshold, direction = best_threshold_accuracy(black, white)
        threshold_rows.append({
            "platform": platform,
            "metric": metric,
            "best_threshold_acc": accuracy,
            "threshold": threshold,
            "direction": direction,
            "black_mean": black.mean(),
            "white_mean": white.mean(),
            "delta": white.mean() - black.mean(),
        })

    features = [metric for metric in CLASSIFIER_METRICS if metric in frame and frame[metric].notna().all()]
    matrix = frame[features].to_numpy()
    target = frame["target"].to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=20260716)
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20260716)),
        "rf200": RandomForestClassifier(n_estimators=200, random_state=20260716, max_features="sqrt"),
    }
    cv_rows = []
    for name, model in models.items():
        scores = cross_val_score(model, matrix, target, cv=splitter, scoring="accuracy")
        cv_rows.append({
            "platform": platform,
            "model": name,
            "features": ";".join(features),
            "cv_acc_mean": scores.mean(),
            "cv_acc_std": scores.std(ddof=1),
            "folds": len(scores),
            "n": len(frame),
        })
    return threshold_rows, cv_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    threshold_rows = []
    cv_rows = []
    all_trials = []
    for platform in PLATFORMS:
        frame = load_trials(root / platform / "data")
        if frame.empty:
            continue
        frame.insert(0, "platform", platform.capitalize())
        all_trials.append(frame)
        platform_thresholds, platform_cv = analyze_platform(platform.capitalize(), frame)
        threshold_rows.extend(platform_thresholds)
        cv_rows.extend(platform_cv)

    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(root / "black_white_all_features.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(root / "black_white_best_threshold_accuracy_all_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(root / "black_white_multifeature_cv.csv", index=False)


if __name__ == "__main__":
    main()
