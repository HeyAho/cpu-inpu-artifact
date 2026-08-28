#!/usr/bin/env python3
"""Compare DummyCNN vs ResNet-50 black/white power results."""
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
base = os.environ.get("INPU_POWER_EXPERIMENT_DIR", str(SCRIPT_DIR))

tests = {
    "DummyCNN (old)": {
        "white": os.path.join(base, "power_test", "ane_power_white.csv"),
        "black": os.path.join(base, "power_test", "ane_power_black.csv"),
    },
    "ResNet-50 (new)": {
        "white": os.path.join(base, "resnet_power_test", "resnet_power_white.csv"),
        "black": os.path.join(base, "resnet_power_test", "resnet_power_black.csv"),
    },
}

results = {}
for model_name, files in tests.items():
    results[model_name] = {}
    for color, path in files.items():
        if not os.path.exists(path):
            print(f"  {model_name} {color}: FILE NOT FOUND")
            continue
        with open(path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = []
            for row in reader:
                try: rows.append([float(v) for v in row])
                except: continue
        if not rows:
            continue
        arr = np.array(rows)
        col_map = {h: i for i, h in enumerate(headers)}
        if "ANE" not in col_map:
            continue
        ane = arr[:, col_map["ANE"]]
        # baseline = first 3s (at 100ms intervals = ~30 samples),
        # active = from 5s to 35s
        t0 = arr[0, 0]
        times = arr[:, 0] - t0
        baseline_mask = times < 3
        active_mask = (times >= 5) & (times < 35)

        if baseline_mask.sum() > 0 and active_mask.sum() > 0:
            b_mean = float(np.mean(ane[baseline_mask]))
            a_mean = float(np.mean(ane[active_mask]))
            delta = a_mean - b_mean
            results[model_name][color] = {
                "baseline": b_mean, "active": a_mean, "delta": delta,
                "times": times.tolist(), "ane": ane.tolist()
            }
            print(f"  {model_name} {color}: baseline={b_mean:.4f}W, "
                  f"active={a_mean:.4f}W, delta={delta:+.4f}W")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Delta comparison
ax = axes[0]
models = list(results.keys())
for i, model in enumerate(models):
    if "white" in results[model] and "black" in results[model]:
        w_delta = results[model]["white"]["delta"]
        b_delta = results[model]["black"]["delta"]
        x = np.array([i - 0.15, i + 0.15])
        ax.bar(x, [w_delta, b_delta], width=0.25,
               color=["white" if model == "DummyCNN (old)" else "salmon",
                      "black" if model == "DummyCNN (old)" else "dimgray"],
               edgecolor="navy", linewidth=1.5,
               label=["White", "Black"] if i == 0 else None)
        ax.text(i - 0.15, w_delta + 0.1, f"{w_delta:.2f}W",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i + 0.15, b_delta + 0.1, f"{b_delta:.2f}W",
                ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace(" (old)", "\n(old)").replace(" (new)", "\n(new)")
                     for m in models])
ax.set_ylabel("ANE Power Delta (W)")
ax.set_title("ANE Power Increase from Baseline", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 2. Ratio bar
ax = axes[1]
ratios = []
if len(results) >= 2:
    for model in models:
        if "white" in results[model] and "black" in results[model]:
            w_d = results[model]["white"]["delta"]
            b_d = results[model]["black"]["delta"]
            ratios.append(w_d / b_d if b_d > 0 else 0)
colors_ratios = ["steelblue", "coral"]
bars = ax.bar(models, ratios, color=colors_ratios[:len(ratios)], edgecolor="navy")
ax.axhline(1.0, color="gray", ls="--", alpha=0.7, label="Equal (1.0x)")
ax.set_ylabel("White/Black Power Ratio")
ax.set_title("White vs Black Power Consumption Ratio", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{ratio:.2f}x", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(ratios) * 1.3 if ratios else 2)

# 3. Time series comparison (ResNet only)
ax = axes[2]
resnet = results.get("ResNet-50 (new)", {})
for color, style, label in [("white", "salmon", "White Input"),
                              ("black", "dimgray", "Black Input")]:
    if color in resnet:
        data = resnet[color]
        times = data["times"]
        ane = data["ane"]
        ax.plot(times, ane, color=style, linewidth=0.6, alpha=0.8, label=label)
        # Add a rolling average
        window = 10
        if len(ane) > window:
            smooth = np.convolve(ane, np.ones(window)/window, mode="valid")
            smooth_t = times[window//2:window//2 + len(smooth)]
            ax.plot(smooth_t, smooth, color="red" if color == "white" else "blue",
                    linewidth=2, alpha=0.7, ls="-")

ax.axvspan(0, 3, alpha=0.1, color="green", label="Baseline")
ax.axvspan(5, 35, alpha=0.1, color="red", label="Inference")
ax.set_xlabel("Time (s)")
ax.set_ylabel("ANE Power (W)")
ax.set_title("ResNet-50: ANE Power Over Time", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("Black vs White Pixel Power: DummyCNN vs ResNet-50",
             fontsize=15, fontweight="bold")
plt.tight_layout()
out = os.path.join(base, "resnet_power_test", "power_comparison.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nPlot -> {out}")
plt.close()
