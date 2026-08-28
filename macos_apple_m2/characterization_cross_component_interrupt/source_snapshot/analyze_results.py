#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ci95(values):
    values = np.asarray(values, dtype=float)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(values), 1.96)
    return critical * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    results = root / "results"; results.mkdir(exist_ok=True)
    intervals = pd.read_csv(root / "data/phase_intervals.csv")
    system = pd.read_csv(root / "data/system_samples.csv")
    monitor_rows = []
    for trial, trial_intervals in intervals.groupby("trial"):
        monitor = pd.read_csv(root / "data/trials" / trial / "monitor.csv")
        for _, interval in trial_intervals.iterrows():
            selected = monitor[(monitor.Timestamp >= interval.start_epoch) & (monitor.Timestamp < interval.end_epoch)].copy()
            if selected.empty:
                continue
            row = interval.to_dict()
            for column in ("ANE", "PCPU", "ECPU", "GPU", "GPU Energy", "GPU_Freq"):
                row[column] = selected[column].mean() if column in selected else np.nan
            row["monitor_rows"] = len(selected)
            monitor_rows.append(row)
    phase_monitor = pd.DataFrame(monitor_rows)
    phase_system = system.groupby(["domain", "target_load_pct", "round", "trial_order", "phase"], as_index=False)[["cpu_total_util_pct", "ane_infer_per_s"]].mean()
    phase = phase_monitor.merge(phase_system, on=["domain", "target_load_pct", "round", "trial_order", "phase"], how="left")
    phase.to_csv(results / "phase_summary.csv", index=False)
    rows = []
    for keys, group in phase.groupby(["domain", "round", "target_load_pct", "trial_order"]):
        domain, round_index, target, order = keys
        indexed = group.set_index("phase")
        pre, load, post = indexed.loc["ane_pre"], indexed.loc["load_on"], indexed.loc["ane_post"]
        row = {"platform": "apple_m2", "workload": "coreml_resnet152_ane", "domain": domain,
               "round": round_index, "target_load_pct": target, "trial_order": order}
        for metric in ("ANE", "PCPU", "ECPU", "GPU", "GPU_Freq", "cpu_total_util_pct", "ane_infer_per_s"):
            row[metric] = load[metric]
            row[f"{metric}_effect"] = load[metric] - (pre[metric] + post[metric]) / 2
        row["actual_load_pct"] = load["cpu_total_util_pct"] if domain == "cpu" else target
        rows.append(row)
    trial = pd.DataFrame(rows).sort_values(["domain", "round", "target_load_pct"])
    trial.to_csv(results / "trial_summary.csv", index=False)
    paired_rows = []
    for (domain, round_index), group in trial.groupby(["domain", "round"]):
        zero = group[group.target_load_pct == 0].iloc[0]
        for _, current in group.iterrows():
            paired_rows.append({
                "domain": domain, "round": round_index, "target_load_pct": current.target_load_pct,
                "actual_load_pct": current.actual_load_pct,
                "ane_power_delta_w": current.ANE - zero.ANE,
                "ane_power_delta_pct": (current.ANE / zero.ANE - 1) * 100,
                "ane_work_delta_pct": (current.ane_infer_per_s / zero.ane_infer_per_s - 1) * 100,
                "gpu_power_delta_w": current.GPU - zero.GPU,
            })
    paired = pd.DataFrame(paired_rows); paired.to_csv(results / "paired_effects.csv", index=False)
    aggregate_rows = []
    report = {"protocol": "cross_platform_ane_v1", "power_source": "IOReport ANE channel"}
    for domain in ("cpu", "gpu"):
        for level, group in trial[trial.domain == domain].groupby("target_load_pct"):
            effects = paired[(paired.domain == domain) & (paired.target_load_pct == level)]
            aggregate_rows.append({
                "platform": "apple_m2", "workload": "coreml_resnet152_ane", "domain": domain,
                "target_load_pct": level, "actual_load_mean_pct": group.actual_load_pct.mean(),
                "ane_power_mean_w": group.ANE.mean(), "ane_power_ci95_w": ci95(group.ANE),
                "ane_power_delta_vs_zero_mean_pct": effects.ane_power_delta_pct.mean(),
                "ane_work_mean_per_s": group.ane_infer_per_s.mean(),
                "ane_work_delta_vs_zero_mean_pct": effects.ane_work_delta_pct.mean(),
                "pcpu_power_mean_w": group.PCPU.mean(), "ecpu_power_mean_w": group.ECPU.mean(),
                "gpu_power_mean_w": group.GPU.mean(), "gpu_freq_mean_mhz": group.GPU_Freq.mean(),
            })
        full = paired[(paired.domain == domain) & (paired.target_load_pct == 100)]
        report[domain] = {
            "full_load_ane_power_delta_mean_w": float(full.ane_power_delta_w.mean()),
            "full_load_ane_power_delta_mean_pct": float(full.ane_power_delta_pct.mean()),
            "full_load_ane_power_delta_ci95_w": float(ci95(full.ane_power_delta_w)),
            "full_load_ane_work_delta_mean_pct": float(full.ane_work_delta_pct.mean()),
        }
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["domain", "target_load_pct"])
    aggregate.to_csv(results / "aggregate_summary.csv", index=False)
    (results / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex="col", constrained_layout=True)
    for column, (domain, color) in enumerate(zip(("cpu", "gpu"), ("#0072B2", "#D55E00"))):
        data = aggregate[aggregate.domain == domain]
        axes[0, column].errorbar(data.target_load_pct, data.ane_power_mean_w, yerr=data.ane_power_ci95_w, marker="o", color=color, capsize=3)
        axes[0, column].set_title(f"{domain.upper()} load")
        axes[1, column].plot(data.target_load_pct, data.ane_work_mean_per_s, marker="s", color=color)
        if domain == "cpu":
            axes[2, column].plot(data.target_load_pct, data.actual_load_mean_pct, marker="o", color=color)
            axes[2, column].set_ylabel("Measured CPU utilization (%)")
        else:
            axes[2, column].plot(data.target_load_pct, data.gpu_power_mean_w, marker="o", color=color)
            axes[2, column].set_ylabel("GPU power (W)")
        for row in range(3): axes[row, column].grid(axis="y", alpha=.3)
        axes[2, column].set_xlabel("Configured load (%)")
    axes[0, 0].set_ylabel("Direct ANE power (W)")
    axes[1, 0].set_ylabel("ResNet152 inference/s")
    fig.suptitle("Apple M2: CPU/GPU load impact on ANE", fontweight="bold")
    fig.savefig(results / "m2_cpu_gpu_to_ane.png", dpi=300)
    fig.savefig(results / "m2_cpu_gpu_to_ane.pdf")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
