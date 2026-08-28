#!/usr/bin/env python3
"""Generate original-style 2-panel figure for EACH repetition."""
import os, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = os.environ.get("INPU_POWER_DATA_DIR", str(SCRIPT_DIR / "data"))
FIGURES_DIR = os.environ.get("INPU_POWER_FIGURES_DIR", str(SCRIPT_DIR / "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Original style
LINE_COLOR = '#324F7D'
FILL_COLOR = '#C5D0E0'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

N_REPS = 5
BASELINE = 5.0
INFERENCE = 20.0

def format_subplot(ax, time_data, power, title, inf_start, inf_end):
    ax.plot(time_data, power, color=LINE_COLOR, linewidth=1.5)
    ax.fill_between(time_data, power, 0, color=FILL_COLOR, alpha=0.8)
    ax.axvline(x=inf_start, color=LINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=inf_end, color=LINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.set_title(title, fontsize=12, pad=10, loc='left')
    ax.set_ylabel('Power (W)', fontsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    mask = (time_data >= inf_start) & (time_data <= inf_end)
    if mask.any():
        avg_pwr = power[mask].mean()
        peak_pwr = power[mask].max()
        ax.text(0.02, 0.85, f'Avg Inference Power: {avg_pwr:.4f} W',
                transform=ax.transAxes, fontsize=11, fontweight='bold', color='#B22222')

def load_data(csv_name):
    path = os.path.join(OUTPUT_DIR, csv_name)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            try: rows.append([float(v) for v in row])
            except: continue
    if not rows:
        return None, None
    arr = np.array(rows)
    col_map = {h:i for i,h in enumerate(headers)}
    if 'ANE' not in col_map:
        return None, None
    t = arr[:, 0] - arr[0, 0]
    return t, arr[:, col_map['ANE']]

for rep in range(N_REPS):
    white_name = f"resnet_power_white_r{rep}.csv"
    black_name = f"resnet_power_black_r{rep}.csv"

    t_w, pwr_w = load_data(white_name)
    t_b, pwr_b = load_data(black_name)

    if t_w is None or t_b is None:
        print(f"Rep {rep}: skip (missing data)")
        continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(f'Apple Neural Engine (ANE) Power Signature — Rep {rep+1}/{N_REPS}',
                 fontsize=16, fontweight='bold', y=0.95)

    # Compute stats for labeling
    mask_w = (t_w >= BASELINE) & (t_w <= BASELINE + INFERENCE)
    mask_b = (t_b >= BASELINE) & (t_b <= BASELINE + INFERENCE)
    avg_w = np.mean(pwr_w[mask_w]) if mask_w.any() else 0
    avg_b = np.mean(pwr_b[mask_b]) if mask_b.any() else 0

    format_subplot(ax1, t_w, pwr_w,
                   f'Dense Compute: White Image (All 1s) — Avg: {avg_w:.4f} W',
                   BASELINE, BASELINE + INFERENCE)

    format_subplot(ax2, t_b, pwr_b,
                   f'Sparse Compute: Black Image (All 0s) — Avg: {avg_b:.4f} W',
                   BASELINE, BASELINE + INFERENCE)

    ax2.set_xlabel('Time (Seconds)', fontsize=10)

    # Add ratio annotation at bottom center
    if avg_b > 0:
        ratio = avg_w / avg_b
        fig.text(0.5, 0.01, f'White/Black Power Ratio: {ratio:.2f}x',
                 ha='center', fontsize=12, fontweight='bold', color='#B22222')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    ts = int(time.time())
    out_path = os.path.join(FIGURES_DIR, f'ane_power_comparison_r{rep}_{ts}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Rep {rep+1}: avg_w={avg_w:.4f}W  avg_b={avg_b:.4f}W  ratio={avg_w/avg_b:.2f}x  -> {out_path}")
    plt.close()

print(f"\nAll {N_REPS} figures saved to {FIGURES_DIR}/")
