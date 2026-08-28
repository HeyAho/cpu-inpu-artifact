#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PLATFORMS = {
    "Intel Meteor Lake": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "AMD XDNA2": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
}


def ci95(values):
    values = np.asarray(values, dtype=float)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(values), 1.96)
    return critical * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def harmonize(intel_path, amd_path):
    intel = pd.read_csv(intel_path)
    intel_common = pd.DataFrame({
        "platform": "Intel Meteor Lake",
        "workload": "OpenVINO ResNet152 inference",
        "domain": intel["domain"],
        "round": intel["round"],
        "target_load_pct": intel["target_load_pct"],
        "actual_load_pct": intel["actual_load_pct"],
        "npu_power_w": intel["npu_power_w"],
        "npu_primary_freq_mhz": intel["npu_primary_freq_mhz"],
        "npu_activity_pct": intel["npu_activity_pct"],
        "npu_work_per_s": intel["npu_work_per_s"],
        "npu_irq_per_s": intel["npu_irq_per_s"],
    })
    amd = pd.read_csv(amd_path)
    if "npu_power_w" in amd.columns:
        amd_common = pd.DataFrame({
            "platform": "AMD XDNA2",
            "workload": "VitisAI ResNet152 INT8 inference",
            "domain": amd["domain"],
            "round": amd["round"],
            "target_load_pct": amd["target_load_pct"],
            "actual_load_pct": amd["actual_load_pct"],
            "npu_power_w": amd["npu_power_w"],
            "npu_primary_freq_mhz": amd["npu_primary_freq_mhz"],
            "npu_activity_pct": amd["npu_activity_pct"],
            "npu_work_per_s": amd["npu_work_per_s"],
            "npu_irq_per_s": amd["npu_irq_per_s"],
        })
    else:
        amd_common = pd.DataFrame({
            "platform": "AMD XDNA2",
            "workload": "XRT-SMI GEMM validation",
            "domain": amd["domain"],
            "round": amd["round"],
            "target_load_pct": amd["target_load_pct"],
            "actual_load_pct": amd["actual_load_pct"],
            "npu_power_w": amd["npu_ipu_power_w"],
            "npu_primary_freq_mhz": amd["npu_ipuclk_mhz"],
            "npu_activity_pct": amd["npu_ipu_activity_mean_pct"],
            "npu_work_per_s": amd["npu_gemm_per_s"],
            "npu_irq_per_s": amd["npu_irq_per_s"],
        })
    return pd.concat([intel_common, amd_common], ignore_index=True)


def normalize_within_round(common):
    frames = []
    for _, group in common.groupby(["platform", "domain", "round"], sort=False):
        zero = group[group["target_load_pct"] == 0].iloc[0]
        group = group.copy()
        for metric in ("npu_power_w", "npu_primary_freq_mhz", "npu_work_per_s", "npu_irq_per_s"):
            group[f"{metric}_normalized_pct"] = group[metric] / zero[metric] * 100.0
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def aggregate(normalized):
    rows = []
    group_columns = ["platform", "workload", "domain", "target_load_pct"]
    metrics = [
        "actual_load_pct", "npu_power_w_normalized_pct", "npu_primary_freq_mhz_normalized_pct",
        "npu_work_per_s_normalized_pct", "npu_irq_per_s_normalized_pct",
    ]
    for keys, group in normalized.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "platform", "target_load_pct"])


def draw(summary, normalized, output_dir):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.8), sharex="col", constrained_layout=True)
    metrics = [
        ("npu_power_w_normalized_pct", "Normalized NPU power (%)"),
        ("npu_primary_freq_mhz_normalized_pct", "Normalized primary frequency (%)"),
        ("npu_work_per_s_normalized_pct", "Normalized work rate (%)"),
        ("npu_irq_per_s_normalized_pct", "Normalized NPU IRQ rate (%)"),
    ]
    for column, domain in enumerate(("cpu", "gpu")):
        for row, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            for platform, style in PLATFORMS.items():
                data = summary[(summary["domain"] == domain) & (summary["platform"] == platform)]
                raw = normalized[(normalized["domain"] == domain) & (normalized["platform"] == platform)]
                platform_offset = -1.0 if platform == "Intel Meteor Lake" else 1.0
                jitter = raw["round"].map({1: -0.7, 2: 0.0, 3: 0.7}).to_numpy()
                axis.scatter(
                    raw["target_load_pct"] + platform_offset + jitter,
                    raw[metric], color=style["color"], marker=style["marker"],
                    s=11, alpha=0.32, linewidths=0, zorder=2,
                )
                axis.errorbar(
                    data["target_load_pct"], data[f"{metric}_mean"],
                    yerr=data[f"{metric}_ci95"], label=platform,
                    color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=1.5, markersize=4, capsize=2.5, capthick=0.8,
                )
            axis.axhline(100, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
            axis.set_ylim(bottom=0)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 0:
                axis.set_title(f"{domain.upper()} load")
            if row == 3:
                axis.set_xlabel("Configured load (%)")
            axis.text(-0.13, 1.04, chr(ord("a") + row * 2 + column), transform=axis.transAxes,
                      fontsize=9, fontweight="bold", va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.025))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_dir / f"cross_platform_normalized_response.{suffix}", bbox_inches="tight")
    fig.savefig(output_dir / "cross_platform_normalized_response.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    image = Image.open(output_dir / "cross_platform_normalized_response.png")
    ImageOps.grayscale(image).save(output_dir / "cross_platform_normalized_response_grayscale.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intel-trial", required=True)
    parser.add_argument("--amd-trial", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = harmonize(args.intel_trial, args.amd_trial)
    normalized = normalize_within_round(common)
    summary = aggregate(normalized)
    common.to_csv(output_dir / "cross_platform_trial_common_schema.csv", index=False)
    normalized.to_csv(output_dir / "cross_platform_trial_normalized.csv", index=False)
    summary.to_csv(output_dir / "cross_platform_aggregate_normalized.csv", index=False)
    draw(summary, normalized, output_dir)


if __name__ == "__main__":
    main()
