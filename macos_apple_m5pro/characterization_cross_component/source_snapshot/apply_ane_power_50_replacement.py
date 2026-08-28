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


def fixed_effect_regression(frame):
    selected = frame[["target_load_pct", "ane_power_w", "round"]].dropna().copy()
    rounds = sorted(selected["round"].unique())
    columns = [np.ones(len(selected)), selected["target_load_pct"].to_numpy(dtype=float)]
    for round_value in rounds[1:]:
        columns.append((selected["round"].to_numpy() == round_value).astype(float))
    design = np.column_stack(columns)
    outcome = selected["ane_power_w"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    degrees_freedom = len(outcome) - design.shape[1]
    residual_variance = float(residual @ residual / degrees_freedom)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    slope_se = float(np.sqrt(max(0.0, covariance[1, 1])))
    slope = float(coefficients[1])
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    p_value = float(2 * stats.t.sf(abs(slope / slope_se), degrees_freedom))
    return {
        "n": len(selected),
        "slope": slope,
        "slope_se": slope_se,
        "ci95_low": slope - critical * slope_se,
        "ci95_high": slope + critical * slope_se,
        "p_value": p_value,
        "degrees_freedom": degrees_freedom,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--replacement-experiment", required=True)
    parser.add_argument("--replace-round", type=int, default=1)
    parser.add_argument("--target-load", type=int, default=50)
    args = parser.parse_args()

    base = Path(args.base_experiment)
    replacement = Path(args.replacement_experiment)
    output = base / "results_corrected_50_replacement"
    output.mkdir(exist_ok=True)

    trials = pd.read_csv(base / "results" / "ane_power_only_trials.csv")
    replacement_trial = pd.read_csv(replacement / "results" / "trial_summary.csv").iloc[0]
    target_mask = (
        trials["round"].eq(args.replace_round)
        & trials["target_load_pct"].eq(args.target_load)
    )
    baseline_mask = (
        trials["round"].eq(args.replace_round)
        & trials["target_load_pct"].eq(0)
    )
    if target_mask.sum() != 1 or baseline_mask.sum() != 1:
        raise RuntimeError("Expected exactly one target row and one same-round baseline row")

    old_row = trials.loc[target_mask].iloc[0].to_dict()
    baseline_w = float(trials.loc[baseline_mask, "ane_power_w"].iloc[0])
    replacement_w = float(replacement_trial["ANE"])
    replacement_norm = replacement_w / baseline_w * 100.0
    trials.loc[target_mask, "ane_power_w"] = replacement_w
    trials.loc[target_mask, "ane_power_norm_pct"] = replacement_norm
    trials.loc[target_mask, "ane_power_delta_pct"] = replacement_norm - 100.0
    trials.to_csv(output / "ane_power_only_trials.csv", index=False)

    rows = []
    for load, group in trials.groupby("target_load_pct", sort=True):
        rows.append({
            "domain": "gpu",
            "target_load_pct": int(load),
            "n": len(group),
            "ane_power_w_mean": float(group["ane_power_w"].mean()),
            "ane_power_w_ci95": ci95(group["ane_power_w"]),
            "ane_power_norm_pct_mean": float(group["ane_power_norm_pct"].mean()),
            "ane_power_norm_pct_ci95": ci95(group["ane_power_norm_pct"]),
            "ane_power_delta_pct_mean": float(group["ane_power_delta_pct"].mean()),
            "ane_power_delta_pct_ci95": ci95(group["ane_power_delta_pct"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "ane_power_only_summary.csv", index=False)

    regression = fixed_effect_regression(trials)
    full = summary.loc[summary.target_load_pct.eq(100)].iloc[0]
    replacement_record = {
        "correction": "replace one pre-identified anomalous 50% ANE power point",
        "raw_base_data_preserved": True,
        "base_experiment": str(base.resolve()),
        "replacement_experiment": str(replacement.resolve()),
        "replaced_condition": {"round": args.replace_round, "target_load_pct": args.target_load},
        "same_round_zero_baseline_w": baseline_w,
        "old_point": {
            "ane_power_w": float(old_row["ane_power_w"]),
            "ane_power_norm_pct": float(old_row["ane_power_norm_pct"]),
        },
        "replacement_point": {
            "ane_power_w": replacement_w,
            "ane_power_norm_pct": replacement_norm,
            "source_trial": str(replacement_trial["trial"]),
        },
        "ane_power_vs_target_load": regression,
        "power_at_100_pct_of_zero_mean": float(full["ane_power_norm_pct_mean"]),
        "power_at_100_pct_of_zero_ci95": float(full["ane_power_norm_pct_ci95"]),
    }
    (output / "replacement_manifest_and_report.json").write_text(
        json.dumps(replacement_record, indent=2, ensure_ascii=False) + "\n"
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.axhline(100, color="0.4", linestyle=":", linewidth=1.4)
    markers = ("o", "^", "D")
    offsets = (-1.2, 0.0, 1.2)
    for plot_index, (round_index, group) in enumerate(trials.groupby("round")):
        ax.scatter(
            group["target_load_pct"] + offsets[plot_index], group["ane_power_norm_pct"],
            facecolors="none", edgecolors="#0072B2", s=48, linewidth=1.4,
            marker=markers[plot_index], label=f"Round {int(round_index)}",
        )
    ax.errorbar(
        summary["target_load_pct"], summary["ane_power_norm_pct_mean"],
        yerr=summary["ane_power_norm_pct_ci95"], color="#0072B2",
        marker="s", markersize=6, linewidth=2, capsize=4, label="Mean +/- 95% CI",
    )
    ax.set_xlabel("Configured GPU load (%)")
    ax.set_ylabel("ANE power (% of same-round 0% baseline)")
    ax.set_title("M5 Pro: corrected ANE power (MPS 4096x4096)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"m5pro_gpu4096_ane_power_corrected.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
