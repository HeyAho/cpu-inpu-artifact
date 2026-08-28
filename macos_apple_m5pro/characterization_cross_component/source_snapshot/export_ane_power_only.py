#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    results = root / "results"
    trials = pd.read_csv(results / "normalized_trials.csv")
    summary = pd.read_csv(results / "normalized_summary.csv")
    trials = trials[trials["domain"] == "gpu"].copy()
    summary = summary[summary["domain"] == "gpu"].copy()

    trial_columns = [
        "platform", "domain", "round", "target_load_pct",
        "ane_power_w", "ane_power_norm_pct", "ane_power_delta_pct",
    ]
    summary_columns = [
        "domain", "target_load_pct", "n",
        "ane_power_w_mean", "ane_power_w_ci95",
        "ane_power_norm_pct_mean", "ane_power_norm_pct_ci95",
        "ane_power_delta_pct_mean", "ane_power_delta_pct_ci95",
    ]
    trials[trial_columns].to_csv(results / "ane_power_only_trials.csv", index=False)
    summary[summary_columns].to_csv(results / "ane_power_only_summary.csv", index=False)

    report = json.loads((results / "report.json").read_text())
    gpu = report["domains"]["gpu"]
    power_report = {
        "platform": report["platform"],
        "trial_count": report["trial_count"],
        "gpu_matrix_size": json.loads((root / "metadata.json").read_text())["gpu_matrix_size"],
        "ane_power_vs_target_load": gpu["ane_power_vs_target_load"],
        "full_vs_zero_ane_power_effect_w": gpu["full_vs_zero_ane_power_effect_w"],
        "power_at_100_pct_of_zero_mean": float(
            summary.loc[summary.target_load_pct == 100, "ane_power_norm_pct_mean"].iloc[0]
        ),
        "power_at_100_pct_of_zero_ci95": float(
            summary.loc[summary.target_load_pct == 100, "ane_power_norm_pct_ci95"].iloc[0]
        ),
    }
    (results / "ane_power_only_report.json").write_text(
        json.dumps(power_report, indent=2, ensure_ascii=False) + "\n"
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.axhline(100, color="0.4", linestyle=":", linewidth=1.4)
    markers = ("o", "^", "D")
    offsets = (-1.2, 0.0, 1.2)
    for plot_index, (round_index, group) in enumerate(trials.groupby("round")):
        ax.scatter(
            group["target_load_pct"] + offsets[plot_index], group["ane_power_norm_pct"],
            facecolors="none", edgecolors="#0072B2", s=48, linewidth=1.4,
            marker=markers[plot_index],
            label=f"Round {int(round_index)}",
        )
    ax.errorbar(
        summary["target_load_pct"], summary["ane_power_norm_pct_mean"],
        yerr=summary["ane_power_norm_pct_ci95"], color="#0072B2",
        marker="s", markersize=6, linewidth=2, capsize=4, label="Mean +/- 95% CI",
    )
    ax.set_xlabel("Configured GPU load (%)")
    ax.set_ylabel("ANE power (% of same-round 0% baseline)")
    ax.set_title("M5 Pro: ANE power under MPS 4096x4096 GPU load")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(results / f"m5pro_gpu4096_ane_power_only.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
