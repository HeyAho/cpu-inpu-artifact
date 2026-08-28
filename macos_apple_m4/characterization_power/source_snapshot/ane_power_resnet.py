#!/usr/bin/env python3
"""
ANE Power: Black vs White Pixel Power Consumption Test using ResNet-50.

A deeper model (ResNet-50) creates higher ANE utilization, making
the power difference between black (zero) and white (non-zero) inputs
more measurable via zero-skipping effects in NPU hardware.

Pipeline:
  1. Convert ResNet-50 to CoreML (if not cached)
  2. Run ANEPowerMonitor while inferencing with black / white inputs
  3. Compare ANE power consumption
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time
import subprocess
import os
import sys
import csv
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Paths
SWIFT_PROBE_PATH = "external/ANEPowerMonitor"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.environ.get("INPU_PYTHON", sys.executable)

MODEL_PATH = os.path.join(OUTPUT_DIR, "ResNet50_ANE.mlpackage")

# Test configuration
INFERENCE_DURATION = 30   # seconds per test
WARMUP_DURATION = 5       # seconds of baseline before inference
COOLDOWN_DURATION = 5     # seconds after inference
BATCH_SIZE = 1
IMAGE_SIZE = 224


def prepare_model():
    """Convert ResNet-50 to CoreML for ANE acceleration."""
    if os.path.exists(MODEL_PATH):
        print(f"  CoreML model exists: {MODEL_PATH}")
        return MODEL_PATH

    print("  Converting ResNet-50 to CoreML (may take a minute)...")
    model = torch.hub.load("pytorch/vision:v0.19.0", "resnet50", weights=None)
    model.eval()

    example_input = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=example_input.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.save(MODEL_PATH)
    print(f"  Saved CoreML model: {MODEL_PATH}")
    return MODEL_PATH


def run_power_test(model_path, input_data, label, csv_name):
    """Run ANEPowerMonitor while inferencing, measure ANE power."""
    csv_path = os.path.join(OUTPUT_DIR, csv_name)

    print(f"\n  Loading model...")
    loaded_model = ct.models.MLModel(model_path)

    print(f"  Starting ANEPowerMonitor probe -> {csv_name}")
    probe = subprocess.Popen(
        [SWIFT_PROBE_PATH, csv_name],
        cwd=OUTPUT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Baseline (idle)
        print(f"  Recording baseline (idle, {WARMUP_DURATION}s)...")
        time.sleep(WARMUP_DURATION)

        # Inference phase
        print(f"  Running {label} inference ({INFERENCE_DURATION}s)...")
        start = time.time()
        iterations = 0
        while time.time() - start < INFERENCE_DURATION:
            loaded_model.predict({"x": input_data})
            iterations += 1

        total_time = time.time() - start
        throughput = iterations / total_time
        print(f"    Completed {iterations} iterations ({throughput:.1f} iter/s)")

        # Cooldown
        print(f"  Recording cooldown ({COOLDOWN_DURATION}s)...")
        time.sleep(COOLDOWN_DURATION)

    finally:
        probe.terminate()
        try:
            probe.wait(timeout=5)
        except subprocess.TimeoutExpired:
            probe.kill()

    # Verify CSV
    if not os.path.exists(csv_path):
        print(f"  WARNING: CSV not generated at {csv_path}")
        return None, 0

    size_kb = os.path.getsize(csv_path) / 1024
    print(f"  CSV saved: {csv_name} ({size_kb:.1f} KB)")

    return csv_path, throughput


def analyze_power_csv(csv_path, label):
    """Parse ANEPowerMonitor CSV and extract power statistics by phase."""
    if csv_path is None or not os.path.exists(csv_path):
        return None

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except (ValueError, IndexError):
                continue

    if not rows:
        return None

    arr = np.array(rows)
    timestamps = arr[:, 0]
    t0 = timestamps[0]
    times = timestamps - t0

    # Find column indices
    col_map = {}
    for i, h in enumerate(headers):
        col_map[h] = i

    # Phase identification
    # baseline: 0-WARMUP_DURATION
    # inference: WARMUP_DURATION to WARMUP_DURATION+INFERENCE_DURATION
    # cooldown: after inference
    t1 = WARMUP_DURATION
    t2 = WARMUP_DURATION + INFERENCE_DURATION

    phases = {
        "baseline":   (0, t1),
        "inference":  (t1, t2),
        "cooldown":   (t2, max(times)),
    }

    results = {"label": label, "throughput": 0}
    for key in ["ANE", "GPU", "GPU_Freq", "GPU Energy"]:
        if key not in col_map:
            continue
        ci = col_map[key]
        for phase_name, (t_start, t_end) in phases.items():
            mask = (times >= t_start) & (times < t_end)
            if mask.sum() == 0:
                continue
            vals = arr[mask, ci]
            results[f"{phase_name}_{key}_mean"] = float(np.mean(vals))
            results[f"{phase_name}_{key}_max"] = float(np.max(vals))
            results[f"{phase_name}_{key}_std"] = float(np.std(vals))

    return results


def plot_comparison(results_list):
    """Generate comparison plots."""
    if not results_list:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = [r["label"] for r in results_list]

    # 1. ANE Power: baseline vs inference
    ax1 = axes[0, 0]
    x = np.arange(len(labels))
    w = 0.3
    baseline_ane = [r.get("baseline_ANE_mean", 0) for r in results_list]
    inference_ane = [r.get("inference_ANE_mean", 0) for r in results_list]
    ax1.bar(x - w/2, baseline_ane, w, label="Baseline (Idle)", color="lightgray")
    ax1.bar(x + w/2, inference_ane, w, label="Inference (Active)", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ANE Power (W)")
    ax1.set_title("ANE Power: Baseline vs Inference", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    for i in range(len(labels)):
        h = inference_ane[i] if i < len(inference_ane) else 0
        b = baseline_ane[i] if i < len(baseline_ane) else 0
        if h > 0:
            ax1.text(i + w/2, h + max(inference_ane)*0.02,
                     f"{h:.4f}W", ha="center", va="bottom", fontsize=9, rotation=90)

    # 2. ANE power increase (delta from baseline)
    ax2 = axes[0, 1]
    deltas = [r.get("inference_ANE_mean", 0) - r.get("baseline_ANE_mean", 0)
              for r in results_list]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
    ax2.bar(labels, deltas, color=colors, edgecolor="navy", alpha=0.8)
    ax2.set_ylabel("ANE Power Delta (W)")
    ax2.set_title("ANE Power Increase During Inference", fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(deltas):
        ax2.text(i, d + max(deltas)*0.02 if d > 0 else d - max(deltas)*0.08,
                 f"{d:.4f}W", ha="center", fontsize=10, fontweight="bold")

    # 3. GPU power comparison
    ax3 = axes[1, 0]
    baseline_gpu = [r.get("baseline_GPU Energy_mean", 0) for r in results_list]
    inference_gpu = [r.get("inference_GPU Energy_mean", 0) for r in results_list]
    if any(baseline_gpu) or any(inference_gpu):
        ax3.bar(x - w/2, baseline_gpu, w, label="Baseline", color="lightgray")
        ax3.bar(x + w/2, inference_gpu, w, label="Inference", color="coral")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("GPU Energy")
        ax3.set_title("GPU Energy During ANE Inference", fontsize=13, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

    # 4. GPU frequency
    ax4 = axes[1, 1]
    inference_freq = [r.get("inference_GPU_Freq_mean", 0) for r in results_list]
    if any(inference_freq):
        ax4.bar(labels, inference_freq, color="mediumpurple", edgecolor="navy", alpha=0.8)
        ax4.set_ylabel("GPU_Freq (MHz)")
        ax4.set_title("GPU Frequency During ANE Inference", fontsize=13, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Black vs White Pixel Power Consumption (ResNet-50 on ANE)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "black_vs_white_power.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\n  Plot saved -> {out_path}")
    plt.close()

    # Time series plot from raw CSV
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    for idx, r in enumerate(results_list[:2]):  # black and white
        csv_path = r.get("_csv_path")
        if not csv_path or not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = []
            for row in reader:
                try:
                    rows.append([float(v) for v in row])
                except:
                    continue
        if not rows:
            continue
        arr = np.array(rows)
        times = arr[:, 0] - arr[0, 0]
        col_map = {h: i for i, h in enumerate(headers)}
        if "ANE" not in col_map:
            continue
        ane_col = col_map["ANE"]
        ax = axes2[idx]
        ax.plot(times, arr[:, ane_col], "b-", alpha=0.7, linewidth=0.8)
        ax.axvspan(0, WARMUP_DURATION, alpha=0.1, color="green", label="Baseline")
        ax.axvspan(WARMUP_DURATION, WARMUP_DURATION + INFERENCE_DURATION,
                   alpha=0.1, color="red", label="Inference")
        ax.axvspan(WARMUP_DURATION + INFERENCE_DURATION, max(times),
                   alpha=0.1, color="gray", label="Cooldown")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ANE Power (W)")
        ax.set_title(f"{r['label']} - ANE Power Time Series", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ANE Power Over Time: Black vs White Input", fontsize=14, fontweight="bold")
    plt.tight_layout()
    ts_path = os.path.join(OUTPUT_DIR, "black_vs_white_timeseries.png")
    plt.savefig(ts_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Time series saved -> {ts_path}")
    plt.close()


def main():
    print("=" * 70)
    print("  Black vs White Pixel Power Test (ResNet-50 on ANE)")
    print("=" * 70)

    # Check ANEPowerMonitor
    if not os.path.exists(SWIFT_PROBE_PATH):
        print(f"ERROR: ANEPowerMonitor not found at {SWIFT_PROBE_PATH}")
        sys.exit(1)

    # Prepare model
    print("\n  [Step 1] Preparing ResNet-50 CoreML model...")
    model_path = prepare_model()

    # Create inputs: black (all zeros) vs white (all ones)
    print("\n  [Step 2] Preparing inputs...")
    black_input = np.zeros((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    white_input = np.ones((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # Run tests
    print("\n  [Step 3] Running power tests...")
    results = []

    # White test first
    print("\n" + "-" * 50)
    print("  TEST: White Input (all 1.0 - dense MAC)")
    print("-" * 50)
    csv_white, tp_white = run_power_test(
        model_path, white_input, "White (Ones)",
        "resnet_power_white.csv"
    )
    r_white = analyze_power_csv(csv_white, "White (Ones)")
    if r_white:
        r_white["_csv_path"] = csv_white
        r_white["throughput"] = tp_white
        results.append(r_white)

    # Cool-down break
    print("\n  Cooling down (15s)...")
    time.sleep(15)

    # Black test
    print("\n" + "-" * 50)
    print("  TEST: Black Input (all 0.0 - zero-skipping)")
    print("-" * 50)
    csv_black, tp_black = run_power_test(
        model_path, black_input, "Black (Zeros)",
        "resnet_power_black.csv"
    )
    r_black = analyze_power_csv(csv_black, "Black (Zeros)")
    if r_black:
        r_black["_csv_path"] = csv_black
        r_black["throughput"] = tp_black
        results.append(r_black)

    # Report
    print("\n" + "=" * 70)
    print("  RESULTS: ANE Power Consumption Comparison")
    print("=" * 70)

    print(f"\n{'Input':<20} {'ANE Baseline':>14} {'ANE Active':>14} {'Delta':>10} "
          f"{'GPU Freq':>10} {'Throughput':>12}")
    print("-" * 80)

    for r in results:
        ane_base = r.get("baseline_ANE_mean", 0)
        ane_active = r.get("inference_ANE_mean", 0)
        delta = ane_active - ane_base
        gpu_freq = r.get("inference_GPU_Freq_mean", 0)
        tp = r.get("throughput", 0)
        print(f"{r['label']:<20} {ane_base:>12.4f}W {ane_active:>12.4f}W "
              f"{delta:>+8.4f}W {gpu_freq:>8.1f}MHz {tp:>8.1f} it/s")

    # Calculate white vs black ratio
    if len(results) >= 2:
        white_delta = results[0].get("inference_ANE_mean", 0) - results[0].get("baseline_ANE_mean", 0)
        black_delta = results[1].get("inference_ANE_mean", 0) - results[1].get("baseline_ANE_mean", 0)
        if black_delta > 0:
            ratio = white_delta / black_delta
            print(f"\n  White/Black power delta ratio: {ratio:.2f}x")
            if ratio > 1.05:
                print("  -> White input consumes MORE power (dense MACs, no zero-skipping)")
            elif ratio < 0.95:
                print("  -> Black input consumes MORE power (unexpected)")
            else:
                print("  -> Black and white consume similar power (small difference)")

    # Plot
    print("\n  [Step 4] Generating plots...")
    plot_comparison(results)

    print("\n" + "=" * 70)
    print("  Power test complete!")
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
