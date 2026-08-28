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

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
})


def ci95(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--full-level", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    results = root / "results"
    trial = pd.read_csv(results / "trial_summary.csv")

    normalized_rows = []
    for (domain, round_index), group in trial.groupby(["domain", "round"]):
        baseline = group[group.target_load_pct == 0]
        if len(baseline) != 1:
            raise ValueError(f"Expected one 0% baseline for {domain} round {round_index}")
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            normalized_rows.append({
                "platform": row.platform,
                "domain": domain,
                "round": int(round_index),
                "target_load_pct": int(row.target_load_pct),
                "ane_power_w": float(row.ANE),
                "ane_power_norm_pct": float(row.ANE / base.ANE * 100.0),
                "ane_power_delta_pct": float((row.ANE / base.ANE - 1.0) * 100.0),
                "ane_throughput_per_s": float(row.ane_infer_per_s),
                "ane_throughput_norm_pct": float(row.ane_infer_per_s / base.ane_infer_per_s * 100.0),
                "ane_throughput_delta_pct": float((row.ane_infer_per_s / base.ane_infer_per_s - 1.0) * 100.0),
                "interference_power": float(row.interference_power),
                "cpu_util_pct": float(row.cpu_total_util_pct),
            })
    normalized = pd.DataFrame(normalized_rows).sort_values(["domain", "round", "target_load_pct"])
    normalized.to_csv(results / "normalized_trials.csv", index=False)

    summary_rows = []
    metrics = (
        "ane_power_w", "ane_power_norm_pct", "ane_power_delta_pct",
        "ane_throughput_per_s", "ane_throughput_norm_pct", "ane_throughput_delta_pct",
        "interference_power", "cpu_util_pct",
    )
    for (domain, level), group in normalized.groupby(["domain", "target_load_pct"]):
        out = {"domain": domain, "target_load_pct": int(level), "n": len(group)}
        for metric in metrics:
            out[f"{metric}_mean"] = float(group[metric].mean())
            out[f"{metric}_ci95"] = ci95(group[metric])
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows).sort_values(["domain", "target_load_pct"])
    summary.to_csv(results / "normalized_summary.csv", index=False)

    domains = tuple(domain for domain in ("cpu", "gpu") if domain in set(normalized.domain))
    report = {"platform": "apple_m5_pro", "rounds": 3, "domains": {}}
    for domain in domains:
        full = summary[(summary.domain == domain) & (summary.target_load_pct == args.full_level)].iloc[0]
        report["domains"][domain] = {
            "power_at_100_pct_of_zero_mean": full.ane_power_norm_pct_mean,
            "power_at_100_pct_of_zero_ci95": full.ane_power_norm_pct_ci95,
            "throughput_at_100_pct_of_zero_mean": full.ane_throughput_norm_pct_mean,
            "throughput_at_100_pct_of_zero_ci95": full.ane_throughput_norm_pct_ci95,
            "interference_power_at_100_mean": full.interference_power_mean,
            "interference_power_at_100_ci95": full.interference_power_ci95,
        }
    (results / "normalized_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    colors = {"cpu": "#0072B2", "gpu": "#D55E00"}
    width = 3.5 * len(domains)
    figure, axes = plt.subplots(2, len(domains), figsize=(width, 4.2), sharex=True, squeeze=False, constrained_layout=True)
    rng = np.random.default_rng(20260713)
    for column, domain in enumerate(domains):
        selected = normalized[normalized.domain == domain]
        for row_index, (metric, ylabel) in enumerate((
            ("ane_power_norm_pct", "ANE power (%)"),
            ("ane_throughput_norm_pct", "ANE throughput (%)"),
        )):
            axis = axes[row_index, column]
            levels = sorted(selected.target_load_pct.unique())
            for level in levels:
                values = selected[selected.target_load_pct == level][metric].to_numpy()
                axis.scatter(
                    np.full(len(values), level) + rng.uniform(-1.7, 1.7, len(values)), values,
                    s=20, facecolors="white", edgecolors=colors[domain], linewidths=0.9, zorder=2,
                )
            means = selected.groupby("target_load_pct")[metric].mean()
            errors = selected.groupby("target_load_pct")[metric].apply(ci95)
            axis.errorbar(
                means.index, means.values, yerr=errors.values, color=colors[domain],
                marker="o" if domain == "cpu" else "s", linewidth=1.4, capsize=2.5, zorder=3,
            )
            axis.axhline(100, color="#666666", linewidth=0.8, linestyle=":")
            axis.grid(axis="y", alpha=0.2)
            if column == 0:
                axis.set_ylabel(ylabel)
        axes[0, column].set_title(f"{domain.upper()} load")
        axes[1, column].set_xlabel("Configured load (%)")
        axes[1, column].set_xticks(sorted(selected.target_load_pct.unique()))
    figure_stem = "m5pro_" + "_".join(domains) + "_ane_normalized"
    figure.savefig(results / f"{figure_stem}.pdf")
    figure.savefig(results / f"{figure_stem}.png", dpi=300)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
