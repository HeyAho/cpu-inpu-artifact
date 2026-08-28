#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "npu_power_w", "npu_primary_freq_mhz", "npu_activity_pct", "npu_work_per_s",
    "npu_irq_per_s", "npu_irq_vectors", "cpu_total_util_pct", "gpu_activity_pct",
    "gpu_freq_mhz", "package_power_w",
]


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * values.std(ddof=1) / np.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(experiment_dir / "data" / "raw_samples.csv")
    selected = raw[raw["phase"].isin(["npu_pre", "load_on", "npu_post"])]
    phase = selected.groupby(
        ["platform", "workload", "domain", "round", "target_load_pct", "trial_order", "phase"],
        as_index=False,
    )[METRICS].mean()
    phase.to_csv(results_dir / "phase_summary.csv", index=False)
    rows = []
    effects_metrics = ["npu_power_w", "npu_primary_freq_mhz", "npu_activity_pct", "npu_work_per_s", "npu_irq_per_s"]
    for keys, group in phase.groupby(["platform", "workload", "domain", "round", "target_load_pct", "trial_order"]):
        platform, workload, domain, round_index, target, order = keys
        indexed = group.set_index("phase")
        pre, load, post = indexed.loc["npu_pre"], indexed.loc["load_on"], indexed.loc["npu_post"]
        row = {
            "platform": platform, "workload": workload, "domain": domain, "round": round_index,
            "target_load_pct": target, "trial_order": order,
            "actual_load_pct": load["cpu_total_util_pct"] if domain == "cpu" else load["gpu_activity_pct"],
        }
        for metric in METRICS:
            row[metric] = load[metric]
        for metric in effects_metrics:
            row[f"{metric}_effect"] = load[metric] - (pre[metric] + post[metric]) / 2.0
        rows.append(row)
    trial = pd.DataFrame(rows).sort_values(["domain", "round", "target_load_pct"])
    trial.to_csv(results_dir / "trial_summary.csv", index=False)
    paired_rows = []
    for (domain, round_index), group in trial.groupby(["domain", "round"]):
        zero = group[group["target_load_pct"] == 0].iloc[0]
        for _, current in group.iterrows():
            paired_rows.append({
                "platform": current["platform"], "workload": current["workload"],
                "domain": domain, "round": round_index, "target_load_pct": current["target_load_pct"],
                "actual_load_pct": current["actual_load_pct"],
                "npu_power_delta_vs_zero_w": current["npu_power_w"] - zero["npu_power_w"],
                "npu_power_delta_vs_zero_pct": (current["npu_power_w"] / zero["npu_power_w"] - 1) * 100,
                "npu_freq_delta_vs_zero_mhz": current["npu_primary_freq_mhz"] - zero["npu_primary_freq_mhz"],
                "npu_irq_delta_vs_zero_pct": (current["npu_irq_per_s"] / zero["npu_irq_per_s"] - 1) * 100,
                "npu_work_delta_vs_zero_pct": (current["npu_work_per_s"] / zero["npu_work_per_s"] - 1) * 100,
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(results_dir / "paired_effects.csv", index=False)
    aggregate_rows = []
    report = {"protocol": "cross_platform_v1", "power_source": "Direct Intel PMT VPU_ENERGY delta"}
    for domain in ("cpu", "gpu"):
        for level, group in trial[trial["domain"] == domain].groupby("target_load_pct"):
            effects = paired[(paired["domain"] == domain) & (paired["target_load_pct"] == level)]
            aggregate_rows.append({
                "platform": group["platform"].iloc[0], "workload": group["workload"].iloc[0],
                "domain": domain, "target_load_pct": level,
                "actual_load_mean_pct": group["actual_load_pct"].mean(),
                "npu_power_mean_w": group["npu_power_w"].mean(),
                "npu_power_ci95_w": ci95(group["npu_power_w"]),
                "npu_power_delta_vs_zero_mean_pct": effects["npu_power_delta_vs_zero_pct"].mean(),
                "npu_primary_freq_mean_mhz": group["npu_primary_freq_mhz"].mean(),
                "npu_activity_mean_pct": group["npu_activity_pct"].mean(),
                "npu_irq_mean_per_s": group["npu_irq_per_s"].mean(),
                "npu_work_mean_per_s": group["npu_work_per_s"].mean(),
                "npu_work_delta_vs_zero_mean_pct": effects["npu_work_delta_vs_zero_pct"].mean(),
            })
        full = paired[(paired["domain"] == domain) & (paired["target_load_pct"] == 100)]
        report[domain] = {
            "full_load_power_delta_vs_zero_mean_w": float(full["npu_power_delta_vs_zero_w"].mean()),
            "full_load_power_delta_vs_zero_mean_pct": float(full["npu_power_delta_vs_zero_pct"].mean()),
            "full_load_power_delta_vs_zero_ci95_w": float(ci95(full["npu_power_delta_vs_zero_w"])),
            "full_load_primary_freq_delta_vs_zero_mean_mhz": float(full["npu_freq_delta_vs_zero_mhz"].mean()),
            "full_load_irq_delta_vs_zero_mean_pct": float(full["npu_irq_delta_vs_zero_pct"].mean()),
            "full_load_work_delta_vs_zero_mean_pct": float(full["npu_work_delta_vs_zero_pct"].mean()),
        }
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["domain", "target_load_pct"])
    aggregate.to_csv(results_dir / "aggregate_summary.csv", index=False)
    (results_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
    for column, (domain, color) in enumerate(zip(("cpu", "gpu"), ("#0072B2", "#D55E00"))):
        data = aggregate[aggregate["domain"] == domain]
        axes[0, column].errorbar(data["target_load_pct"], data["npu_power_mean_w"], yerr=data["npu_power_ci95_w"], marker="o", linewidth=2, capsize=4, color=color)
        axes[0, column].set_title(f"{domain.upper()} load")
        axes[0, column].grid(True, linestyle="--", alpha=0.35)
        axes[1, column].plot(data["target_load_pct"], data["npu_primary_freq_mean_mhz"], marker="o", color=color)
        axes[1, column].grid(True, linestyle="--", alpha=0.35)
        axes[2, column].plot(data["target_load_pct"], data["npu_irq_mean_per_s"], marker="o", color=color, label="IRQ/s")
        throughput = axes[2, column].twinx()
        throughput.plot(data["target_load_pct"], data["npu_work_mean_per_s"], marker="s", linestyle="--", color="#009E73", label="Inference/s")
        left, left_labels = axes[2, column].get_legend_handles_labels()
        right, right_labels = throughput.get_legend_handles_labels()
        axes[2, column].legend(left + right, left_labels + right_labels, loc="lower left")
        axes[2, column].set_xlabel("Configured load (%)")
        axes[2, column].grid(True, linestyle="--", alpha=0.35)
        throughput.set_ylabel("NPU inference/s")
    axes[0, 0].set_ylabel("Direct NPU power (W)")
    axes[1, 0].set_ylabel("Primary NPU frequency (MHz)")
    axes[2, 0].set_ylabel("NPU IRQ/s")
    fig.suptitle("Intel NPU response under cross-platform aligned protocol", fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "intel_aligned_power_freq_irq_vs_load.png", dpi=200)
    fig.savefig(results_dir / "intel_aligned_power_freq_irq_vs_load.pdf")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
