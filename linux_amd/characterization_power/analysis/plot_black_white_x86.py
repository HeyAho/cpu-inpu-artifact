#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("assets/black_white_resnet50/Intel_AMD_black_white_resnet50_20260716")
BLACK_COLOR = "#333333"
WHITE_COLOR = "#E69F00"


def load_points(data_dir):
    rows = []
    for label in ["black", "white"]:
        paths = sorted(
            data_dir.glob(f"resnet_power_{label}_r*.csv"),
            key=lambda path: int(path.stem.rsplit("r", 1)[1]),
        )
        for path in paths:
            frame = pd.read_csv(path)
            infer = frame[frame["phase"] == "infer"].copy()
            if infer.empty:
                continue
            rows.append({
                "label": label,
                "rep": int(path.stem.rsplit("r", 1)[1]),
                "npu_power_w": infer["npu_power_w"].mean(),
                "npu_freq_mhz": infer["npu_freq_mhz"].mean() if "npu_freq_mhz" in infer else np.nan,
                "npu_activity_pct": infer["npu_activity_pct"].mean() if "npu_activity_pct" in infer else np.nan,
                "npu_work_per_s": infer["npu_work_per_s"].mean() if "npu_work_per_s" in infer else np.nan,
                "samples": len(infer),
            })
    return pd.DataFrame(rows)


def summarize(frame):
    rows = []
    for metric in ["npu_power_w", "npu_freq_mhz", "npu_activity_pct", "npu_work_per_s"]:
        black = frame[frame["label"] == "black"][metric].dropna().to_numpy()
        white = frame[frame["label"] == "white"][metric].dropna().to_numpy()
        if len(black) == 0 or len(white) == 0:
            continue
        if white.mean() >= black.mean():
            ordering_errors = sum(
                1 for black_value in black for white_value in white if white_value <= black_value
            )
            direction = "white_high"
        else:
            ordering_errors = sum(
                1 for black_value in black for white_value in white if white_value >= black_value
            )
            direction = "white_low"
        rows.append({
            "metric": metric,
            "black_mean": black.mean(),
            "black_std": black.std(ddof=1) if len(black) > 1 else 0.0,
            "white_mean": white.mean(),
            "white_std": white.std(ddof=1) if len(white) > 1 else 0.0,
            "delta": white.mean() - black.mean(),
            "ratio": white.mean() / black.mean() if black.mean() else np.nan,
            "direction": direction,
            "pairwise_ordering_errors": ordering_errors,
            "pairs": len(black) * len(white),
            "black_n": len(black),
            "white_n": len(white),
        })
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


def plot_platform(axis, platform, frame):
    black = frame[frame["label"] == "black"].sort_values("rep")
    white = frame[frame["label"] == "white"].sort_values("rep")
    axis.scatter(
        black["rep"], black["npu_power_w"], s=52, marker="s", c=BLACK_COLOR,
        edgecolors="black", linewidth=0.5, label=f"Black (n={len(black)})",
    )
    axis.scatter(
        white["rep"], white["npu_power_w"], s=52, marker="o", c=WHITE_COLOR,
        edgecolors="black", linewidth=0.5, label=f"White (n={len(white)})",
    )
    black_values = black["npu_power_w"].to_numpy()
    white_values = white["npu_power_w"].to_numpy()
    black_mean = black_values.mean()
    white_mean = white_values.mean()
    if white_mean >= black_mean:
        ordering_errors = sum(
            1 for black_value in black_values for white_value in white_values if white_value <= black_value
        )
    else:
        ordering_errors = sum(
            1 for black_value in black_values for white_value in white_values if white_value >= black_value
        )
    accuracy, _, _ = best_threshold_accuracy(black_values, white_values)
    axis.axhline(black_mean, color=BLACK_COLOR, linestyle="--", linewidth=1.8, alpha=0.65)
    axis.axhline(white_mean, color=WHITE_COLOR, linestyle="--", linewidth=1.8, alpha=0.75)
    axis.set_title(
        f"{platform}: Δ={white_mean - black_mean:+.3f} W, "
        f"ordering errors={ordering_errors}/{len(black_values) * len(white_values)}, "
        f"best acc.={accuracy:.1%}",
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xlabel("Repetition")
    axis.set_ylabel("NPU mean power (W)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=9, frameon=True)


def save_figure(figure, root, stem):
    figure.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(root / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root
    datasets = [
        ("Intel", root / "intel" / "data"),
        ("AMD", root / "amd" / "data"),
    ]
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    frames = []
    threshold_rows = []
    usable = []

    for platform, data_dir in datasets:
        frame = load_points(data_dir)
        if frame.empty:
            continue
        frame.insert(0, "platform", platform)
        frames.append(frame)
        usable.append((platform, frame))

        summary = summarize(frame)
        summary.insert(0, "platform", platform)
        summaries.append(summary)
        summary.to_csv(root / f"{platform.lower()}_black_white_summary.csv", index=False)

        black = frame[frame["label"] == "black"]["npu_power_w"].dropna().to_numpy()
        white = frame[frame["label"] == "white"]["npu_power_w"].dropna().to_numpy()
        accuracy, threshold, direction = best_threshold_accuracy(black, white)
        threshold_rows.append({
            "platform": platform,
            "metric": "npu_power_w",
            "best_threshold_acc": accuracy,
            "threshold": threshold,
            "direction": direction,
        })

        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        plot_platform(axis, platform, frame)
        figure.tight_layout()
        save_figure(figure, root, f"{platform.lower()}_black_white_scatter")
        plt.close(figure)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root / "black_white_points.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(root / "black_white_summary.csv", index=False)
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(root / "black_white_best_threshold_accuracy.csv", index=False)

    if usable:
        figure, axes = plt.subplots(1, len(usable), figsize=(7.2 * len(usable), 4.4), squeeze=False)
        for axis, (platform, frame) in zip(axes[0], usable):
            plot_platform(axis, platform, frame)
        figure.suptitle("Black vs White Inputs on x86 NPU (Aligned ResNet50, B=1)", y=1.03,
                       fontsize=14, fontweight="bold")
        figure.tight_layout()
        save_figure(figure, root, "intel_amd_black_white_scatter")
        plt.close(figure)


if __name__ == "__main__":
    main()
