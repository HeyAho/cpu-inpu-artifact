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


POWER_COLUMNS = (
    "ANE", "GPU", "PCPU", "ECPU",
    "CPU_Super", "CPU_Performance_0", "CPU_Performance_1",
    "CPU_Performance", "CPU_Total",
)
EFFECT_METRICS = (
    "ANE",
    "cpu_super_power",
    "cpu_performance_power",
    "cpu_domain_power",
    "GPU",
    "cpu_total_util_pct",
    "ane_infer_per_s",
    "ane_irq_per_s",
    "ane_irq_per_inference",
)


def ci95(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def percent_change(value, baseline):
    if pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return (value / baseline - 1.0) * 100.0


def irq_phase_rows(root, intervals, channel_name):
    rows = []
    for trial_name, trial_intervals in intervals.groupby("trial"):
        irq_path = root / "data" / "trials" / trial_name / "irq.csv"
        if not irq_path.exists():
            continue
        irq = pd.read_csv(irq_path)
        irq["channel_norm"] = irq["channel"].astype(str).str.strip()
        selected_channel = irq[irq.channel_norm == channel_name]
        if selected_channel.empty:
            continue
        for _, interval in trial_intervals.iterrows():
            selected = selected_channel[
                (selected_channel.timestamp >= interval.start_epoch)
                & (selected_channel.timestamp < interval.end_epoch)
            ]
            duration = interval.end_epoch - interval.start_epoch
            row = interval.to_dict()
            row["ane_irq_count"] = selected.value.sum()
            row["ane_irq_per_s"] = selected.value.sum() / duration if duration > 0 else np.nan
            row["irq_samples"] = len(selected)
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_effect_regression(frame, x_column, y_column):
    selected = frame[[x_column, y_column, "round"]].dropna().copy()
    rounds = sorted(selected["round"].unique())
    if len(selected) < 4 or selected[x_column].nunique() < 2:
        return {"n": len(selected), "slope": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan}
    columns = [np.ones(len(selected)), selected[x_column].to_numpy(dtype=float)]
    for round_value in rounds[1:]:
        columns.append((selected["round"].to_numpy() == round_value).astype(float))
    design = np.column_stack(columns)
    outcome = selected[y_column].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    degrees_freedom = len(outcome) - design.shape[1]
    if degrees_freedom <= 0:
        return {"n": len(selected), "slope": float(coefficients[1]), "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan}
    residual_variance = float(residual @ residual / degrees_freedom)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    slope_se = float(np.sqrt(max(0.0, covariance[1, 1])))
    slope = float(coefficients[1])
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    t_value = slope / slope_se if slope_se > 0 else np.inf
    p_value = float(2 * stats.t.sf(abs(t_value), degrees_freedom))
    return {
        "n": len(selected),
        "slope": slope,
        "slope_se": slope_se,
        "ci95_low": slope - critical * slope_se,
        "ci95_high": slope + critical * slope_se,
        "p_value": p_value,
        "degrees_freedom": degrees_freedom,
    }


def classify_trend(regression):
    slope = regression.get("slope", np.nan)
    p_value = regression.get("p_value", np.nan)
    if pd.isna(slope):
        return "insufficient_data"
    if slope < 0 and p_value < 0.05:
        return "significant_decrease"
    if slope < 0:
        return "directional_decrease"
    if slope > 0 and p_value < 0.05:
        return "significant_increase"
    return "no_clear_decrease"


def paired_full_effect(trial, domain, metric):
    rows = []
    for round_index, group in trial[trial.domain == domain].groupby("round"):
        zero = group[group.target_load_pct == 0]
        full = group[group.target_load_pct == 100]
        if zero.empty or full.empty:
            continue
        rows.append(float(full.iloc[0][metric] - zero.iloc[0][metric]))
    return {
        "n": len(rows),
        "mean": float(np.mean(rows)) if rows else np.nan,
        "ci95_half_width": ci95(rows),
        "values": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--irq-channel", default="")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--results-dir-name", default="results")
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    results = root / args.results_dir_name
    results.mkdir(exist_ok=True)
    intervals = pd.read_csv(root / "data" / "phase_intervals.csv")
    system = pd.read_csv(root / "data" / "system_samples.csv")
    ordered_trials = (
        intervals[["trial", "round", "trial_order"]]
        .drop_duplicates()
        .sort_values(["round", "trial_order"])
    )
    if args.max_trials is not None:
        ordered_trials = ordered_trials.head(args.max_trials)
        selected_trials = set(ordered_trials["trial"])
        intervals = intervals[intervals.trial.isin(selected_trials)].copy()
        system = system[system.trial.isin(selected_trials)].copy()

    monitor_rows = []
    for trial_name, trial_intervals in intervals.groupby("trial"):
        monitor_path = root / "data" / "trials" / trial_name / "monitor.csv"
        monitor = pd.read_csv(monitor_path)
        for _, interval in trial_intervals.iterrows():
            selected = monitor[
                (monitor.Timestamp >= interval.start_epoch)
                & (monitor.Timestamp < interval.end_epoch)
            ]
            if selected.empty:
                continue
            row = interval.to_dict()
            for column in POWER_COLUMNS:
                row[column] = selected[column].mean() if column in selected else np.nan
            row["monitor_rows"] = len(selected)
            monitor_rows.append(row)
    phase_monitor = pd.DataFrame(monitor_rows)

    phase_system = system.groupby(
        ["trial", "domain", "target_load_pct", "round", "trial_order", "phase"],
        as_index=False,
    )[["cpu_total_util_pct", "ane_infer_per_s"]].mean()
    merge_columns = ["trial", "domain", "target_load_pct", "round", "trial_order", "phase"]
    phase = phase_monitor.merge(phase_system, on=merge_columns, how="left")
    phase_irq = irq_phase_rows(root, intervals, args.irq_channel) if args.irq_channel else pd.DataFrame()
    if not phase_irq.empty:
        phase = phase.merge(
            phase_irq[["trial", "phase", "ane_irq_count", "ane_irq_per_s", "irq_samples"]],
            on=["trial", "phase"],
            how="left",
        )
    else:
        phase["ane_irq_count"] = np.nan
        phase["ane_irq_per_s"] = np.nan
        phase["irq_samples"] = 0
    phase["cpu_super_power"] = phase["CPU_Super"].combine_first(phase["PCPU"])
    phase["cpu_performance_power"] = phase["CPU_Performance"].combine_first(phase["ECPU"])
    measured_total = phase["CPU_Total"]
    component_total = phase[["cpu_super_power", "cpu_performance_power"]].sum(axis=1, min_count=1)
    phase["cpu_domain_power"] = measured_total.combine_first(component_total)
    phase["ane_irq_per_inference"] = phase["ane_irq_per_s"] / phase["ane_infer_per_s"]
    phase.to_csv(results / "phase_summary.csv", index=False)

    trial_rows = []
    for keys, group in phase.groupby(["trial", "domain", "round", "target_load_pct", "trial_order"]):
        trial_name, domain, round_index, target, order = keys
        indexed = group.set_index("phase")
        required = {"ane_pre", "load_on", "ane_post"}
        if not required.issubset(indexed.index):
            continue
        pre = indexed.loc["ane_pre"]
        load = indexed.loc["load_on"]
        post = indexed.loc["ane_post"]
        row = {
            "platform": args.platform,
            "workload": "coreml_resnet152_ane_saturated",
            "trial": trial_name,
            "domain": domain,
            "round": round_index,
            "target_load_pct": target,
            "trial_order": order,
            "monitor_rows": load["monitor_rows"],
            "irq_samples": load["irq_samples"],
        }
        for metric in EFFECT_METRICS:
            baseline = (pre[metric] + post[metric]) / 2.0
            row[metric] = load[metric]
            row[f"{metric}_baseline"] = baseline
            row[f"{metric}_effect"] = load[metric] - baseline
            row[f"{metric}_effect_pct"] = percent_change(load[metric], baseline)
        row["interference_power"] = row["cpu_domain_power"] if domain == "cpu" else row["GPU"]
        row["interference_power_effect"] = row["cpu_domain_power_effect"] if domain == "cpu" else row["GPU_effect"]
        trial_rows.append(row)
    trial = pd.DataFrame(trial_rows).sort_values(["round", "trial_order"])

    corrected_metrics = (
        "ANE_effect",
        "ANE_effect_pct",
        "ane_infer_per_s_effect_pct",
        "ane_irq_per_s_effect_pct",
        "ane_irq_per_inference_effect_pct",
        "interference_power_effect",
        "cpu_total_util_pct_effect",
    )
    for metric in corrected_metrics:
        trial[f"{metric}_vs_control"] = np.nan
    for (domain, round_index), group in trial.groupby(["domain", "round"]):
        zero = group[group.target_load_pct == 0]
        if zero.empty:
            continue
        zero_row = zero.iloc[0]
        for row_index in group.index:
            for metric in corrected_metrics:
                trial.loc[row_index, f"{metric}_vs_control"] = trial.loc[row_index, metric] - zero_row[metric]
    trial.to_csv(results / "trial_summary.csv", index=False)

    aggregate_rows = []
    aggregate_metrics = (
        "interference_power_effect",
        "cpu_total_util_pct_effect",
        "ANE_effect",
        "ANE_effect_pct",
        "ane_infer_per_s_effect_pct",
        "ane_irq_per_s_effect_pct",
        "ane_irq_per_inference_effect_pct",
    )
    for (domain, level), group in trial.groupby(["domain", "target_load_pct"]):
        row = {"platform": args.platform, "domain": domain, "target_load_pct": level, "n": len(group)}
        for metric in aggregate_metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = ci95(group[metric])
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["domain", "target_load_pct"])
    aggregate.to_csv(results / "aggregate_summary.csv", index=False)

    report = {
        "protocol": "apple_ane_real_compute_v2",
        "platform": args.platform,
        "selection": {
            "max_trials": args.max_trials,
            "ordering": "round, then trial_order",
            "selected_rounds": sorted(int(value) for value in ordered_trials["round"].unique()),
        },
        "trial_count": int(len(trial)),
        "expected_trial_count": int(len(intervals[["trial"]].drop_duplicates())),
        "data_quality": {
            "missing_ane_power": int(trial["ANE"].isna().sum()),
            "missing_irq_rate": int(trial["ane_irq_per_s"].isna().sum()),
            "median_monitor_rows_load_on": float(trial["monitor_rows"].median()),
            "median_irq_samples_load_on": float(trial["irq_samples"].median()),
            "irq_collected": bool(args.irq_channel and not phase_irq.empty),
        },
        "domains": {},
    }
    for domain in ("cpu", "gpu"):
        selected = trial[trial.domain == domain]
        power_regression = fixed_effect_regression(selected, "target_load_pct", "ANE_effect")
        irq_regression = fixed_effect_regression(selected, "target_load_pct", "ane_irq_per_s_effect_pct")
        throughput_regression = fixed_effect_regression(selected, "target_load_pct", "ane_infer_per_s_effect_pct")
        measured_power_regression = fixed_effect_regression(selected, "interference_power_effect", "ANE_effect")
        report["domains"][domain] = {
            "ane_power_vs_target_load": power_regression,
            "ane_irq_vs_target_load": irq_regression,
            "ane_throughput_vs_target_load": throughput_regression,
            "ane_power_vs_measured_interference_power": measured_power_regression,
            "power_trend": classify_trend(power_regression),
            "irq_trend": classify_trend(irq_regression),
            "joint_hypothesis_supported": (
                classify_trend(power_regression) == "significant_decrease"
                and classify_trend(irq_regression) == "significant_decrease"
            ),
            "full_vs_zero_ane_power_effect_w": paired_full_effect(trial, domain, "ANE_effect"),
            "full_vs_zero_ane_irq_effect_pct": paired_full_effect(trial, domain, "ane_irq_per_s_effect_pct"),
            "full_vs_zero_ane_throughput_effect_pct": paired_full_effect(trial, domain, "ane_infer_per_s_effect_pct"),
            "full_vs_zero_interference_power_effect_w": paired_full_effect(trial, domain, "interference_power_effect"),
        }
    (results / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    colors = {"cpu": "#0072B2", "gpu": "#D55E00"}
    metrics = [
        ("interference_power_effect", "Interference-domain power effect (W)"),
        ("ANE_effect", "ANE power effect (W)"),
        ("ane_infer_per_s_effect_pct", "ANE throughput effect (%)"),
        ("ane_irq_per_s_effect_pct", "ANE interrupt-rate effect (%)"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(10, 12), sharex="col", constrained_layout=True)
    rng = np.random.default_rng(20260713)
    for column, domain in enumerate(("cpu", "gpu")):
        domain_data = trial[trial.domain == domain]
        for row_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column]
            for level in sorted(domain_data.target_load_pct.unique()):
                values = domain_data[domain_data.target_load_pct == level][metric].dropna().to_numpy()
                jitter = rng.uniform(-2.2, 2.2, size=len(values))
                axis.scatter(
                    np.full(len(values), level) + jitter,
                    values,
                    s=24,
                    facecolors="white",
                    edgecolors=colors[domain],
                    linewidths=1.0,
                    alpha=0.9,
                    zorder=2,
                )
            means = domain_data.groupby("target_load_pct")[metric].mean()
            errors = domain_data.groupby("target_load_pct")[metric].apply(ci95)
            axis.errorbar(
                means.index,
                means.values,
                yerr=errors.values,
                color=colors[domain],
                marker="o" if domain == "cpu" else "s",
                linestyle="-" if domain == "cpu" else "--",
                capsize=3,
                linewidth=1.6,
                zorder=3,
            )
            axis.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
            axis.grid(axis="y", alpha=0.25)
            if column == 0:
                axis.set_ylabel(label)
        axes[0, column].set_title(f"{domain.upper()} SGEMM interference")
        axes[-1, column].set_xlabel("Configured compute duty cycle (%)")
    display_platform = args.platform.replace("apple_", "Apple ").upper()
    figure.suptitle(f"{display_platform}: real compute load impact on saturated ANE", fontweight="bold")
    figure.savefig(results / f"{args.platform}_real_compute_ane_effects.png", dpi=300)
    figure.savefig(results / f"{args.platform}_real_compute_ane_effects.pdf")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
