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
    "super": "#CC79A7",
    "performance": "#009E73",
    "total": "#333333",
    "util": "#56B4E9",
}


def ci95(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def summarize(frame, metric):
    grouped = frame.groupby("target_load_pct")[metric]
    return grouped.mean(), grouped.apply(ci95)


def draw_raw_and_mean(axis, frame, metric, color, marker, ylabel, seed):
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
    means, errors = summarize(frame, metric)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trial = pd.read_csv(root / "results" / "trial_summary.csv")
    normalized = pd.read_csv(root / "results" / "normalized_trials.csv")

    power_columns = ["cpu_super_power", "cpu_performance_power", "cpu_domain_power"]
    if trial[power_columns].isna().any().any():
        raise ValueError("Corrected Super/Performance/Total CPU power channels are incomplete")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    draw_raw_and_mean(
        axes[0, 0], normalized, "ane_power_norm_pct", COLORS["ane"], "o",
        "ANE power (% of baseline)", 20260713,
    )
    draw_raw_and_mean(
        axes[0, 1], normalized, "ane_throughput_norm_pct", COLORS["throughput"], "s",
        "ANE throughput (% of baseline)", 20260714,
    )
    for axis in axes[0]:
        axis.axhline(100, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_ylim(0, 110)

    power_specs = (
        ("cpu_super_power", "Super", COLORS["super"], "o", "-"),
        ("cpu_performance_power", "Performance", COLORS["performance"], "s", "--"),
        ("cpu_domain_power", "Total", COLORS["total"], "^", "-."),
    )
    for metric, label, color, marker, linestyle in power_specs:
        means, errors = summarize(trial, metric)
        axes[1, 0].errorbar(
            means.index, means.values, yerr=errors.values,
            color=color, marker=marker, linestyle=linestyle,
            linewidth=1.3, markersize=4.2, capsize=2.3, label=label,
        )
    axes[1, 0].set_ylabel("CPU power (reported units)")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend(frameon=False, loc="upper left")

    util_mean, util_error = summarize(trial, "cpu_total_util_pct")
    axes[1, 1].errorbar(
        util_mean.index, util_mean.values, yerr=util_error.values,
        color=COLORS["util"], marker="D", linewidth=1.4,
        markersize=4.2, capsize=2.3, label="Measured",
    )
    axes[1, 1].plot(
        [0, 25, 50, 75, 100], [0, 25, 50, 75, 100],
        color="#777777", linestyle="--", linewidth=1.0, label="Configured",
    )
    axes[1, 1].set_ylabel("CPU utilization (%)")
    axes[1, 1].set_ylim(0, 110)
    axes[1, 1].legend(frameon=False, loc="upper left")

    panel_titles = (
        "(a) ANE power", "(b) ANE throughput",
        "(c) CPU power domains", "(d) Load validation",
    )
    for axis, title in zip(axes.flat, panel_titles):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(-3, 103)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    for axis in axes[1]:
        axis.set_xlabel("Configured CPU load (%)")

    stem = output / "m5pro_corrected_cpu_ane_impact_lines"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)

    gray = plt.imread(stem.with_suffix(".png"))
    luminance = 0.2126 * gray[..., 0] + 0.7152 * gray[..., 1] + 0.0722 * gray[..., 2]
    plt.imsave(output / "m5pro_corrected_cpu_ane_impact_lines_grayscale.png", luminance, cmap="gray")


if __name__ == "__main__":
    main()
