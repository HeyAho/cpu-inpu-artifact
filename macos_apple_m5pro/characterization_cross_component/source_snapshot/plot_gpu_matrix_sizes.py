#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def ci95(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = pd.read_csv(root / "results" / "normalized_trials.csv")
    trial = pd.read_csv(root / "results" / "trial_summary.csv")
    metadata = json.loads((root / "metadata.json").read_text())
    mapping = {item["level"]: item.get("gpu_matrix_size", 0) for item in metadata["schedule"]}
    normalized["matrix_size"] = normalized["target_load_pct"].map(mapping)
    trial["matrix_size"] = trial["target_load_pct"].map(mapping)
    normalized = normalized.sort_values("matrix_size")
    trial = trial.sort_values("matrix_size")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    specs = [
        ("ane_power_delta_pct", "ANE power change (%)", "o", "#0072B2", (-2.5, 2.5), normalized),
        ("ane_throughput_delta_pct", "ANE throughput change (%)", "s", "#D55E00", (-1.0, 1.0), normalized),
        ("GPU", "GPU power (reported units)", "^", "#009E73", (0, max(175.0, float(trial.GPU.max()) * 1.12)), trial),
    ]
    for axis, (metric, ylabel, marker, color, ylim, frame) in zip(axes, specs):
        means = frame.groupby("matrix_size")[metric].mean()
        errors = frame.groupby("matrix_size")[metric].apply(ci95)
        rng = np.random.default_rng(20260713)
        for size in means.index:
            values = frame.loc[frame.matrix_size == size, metric].to_numpy()
            axis.scatter(np.full(len(values), size) + rng.uniform(-18, 18, len(values)), values,
                         s=22, facecolors="white", edgecolors=color, linewidths=0.9, zorder=2)
        axis.errorbar(means.index, means.values, yerr=errors.values, color=color, marker=marker,
                      markersize=4.5, linewidth=1.4, capsize=2.5, zorder=3)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("MPS FP32 matrix size")
        axis.set_xticks(sorted(means.index))
        axis.set_xticklabels(["idle" if x == 0 else f"{int(x)}" for x in sorted(means.index)], rotation=35, ha="right")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axes[0].axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    axes[1].axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    for axis, title in zip(axes, ("(a) ANE power", "(b) ANE throughput", "(c) GPU load validation")):
        axis.set_title(title, loc="left", fontweight="bold")
    stem = output / "m5pro_gpu_matrix_size_ane_impact_lines"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
