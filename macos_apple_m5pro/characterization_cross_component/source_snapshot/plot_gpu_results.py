#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


COLORS = {
    "ane": "#0072B2",
    "throughput": "#D55E00",
    "gpu": "#009E73",
}


def ci95(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def draw(axis, frame, metric, color, marker, ylabel, seed, ylim):
    rng = np.random.default_rng(seed)
    for level in (0, 25, 50, 75, 100):
        values = frame.loc[frame.target_load_pct == level, metric].to_numpy()
        axis.scatter(
            np.full(len(values), level) + rng.uniform(-1.4, 1.4, len(values)),
            values,
            s=22,
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            zorder=2,
        )
    grouped = frame.groupby("target_load_pct")[metric]
    means = grouped.mean()
    errors = grouped.apply(ci95)
    axis.errorbar(
        means.index,
        means.values,
        yerr=errors.values,
        color=color,
        marker=marker,
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        zorder=3,
    )
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="m5pro_gpu_ane_impact_lines")
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = pd.read_csv(root / "results" / "normalized_trials.csv")
    trial = pd.read_csv(root / "results" / "trial_summary.csv")
    normalized = normalized[normalized.domain == "gpu"].copy()
    trial = trial[trial.domain == "gpu"].copy()
    if len(normalized) != 15 or len(trial) != 15:
        raise ValueError("Expected 15 validated GPU trials")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    draw(
        axes[0], normalized, "ane_power_delta_pct", COLORS["ane"], "o",
        "ANE power change (%)", 20260713, (-2.5, 2.5),
    )
    draw(
        axes[1], normalized, "ane_throughput_delta_pct", COLORS["throughput"], "s",
        "ANE throughput change (%)", 20260714, (-1.0, 1.0),
    )
    gpu_upper = max(175.0, float(trial.GPU.max()) * 1.12)
    draw(
        axes[2], trial, "GPU", COLORS["gpu"], "^",
        "GPU power (reported units)", 20260715, (0, gpu_upper),
    )
    for axis in axes[:2]:
        axis.axhline(0, color="#666666", linewidth=0.8, linestyle=":")

    titles = ("(a) ANE power", "(b) ANE throughput", "(c) GPU load validation")
    for axis, title in zip(axes, titles):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Configured GPU load (%)")
        axis.set_xlim(-3, 103)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)

    stem = output / args.output_stem
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)

    image = plt.imread(stem.with_suffix(".png"))
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    plt.imsave(output / f"{args.output_stem}_grayscale.png", luminance, cmap="gray")


if __name__ == "__main__":
    main()
