#!/usr/bin/env python3
"""Generate original-style 2-panel figure from the rep closest to mean."""
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

LINE_COLOR = '#324F7D'
FILL_COLOR = '#C5D0E0'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

N_REPS = 5
BASELINE = 5
INFERENCE = 20

# Find rep closest to mean
def load_ane_stats(csv_name):
    path = os.path.join(OUTPUT_DIR, csv_name)
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            try: rows.append([float(v) for v in row])
            except: continue
    if not rows:
        return None, None, None
    arr = np.array(rows)
    col_map = {h:i for i,h in enumerate(headers)}
    if 'ANE' not in col_map:
        return None, None, None
    t = arr[:, 0] - arr[0, 0]
    ane = arr[:, col_map['ANE']]
    mask = (t >= BASELINE) & (t <= BASELINE + INFERENCE)
    delta = float(np.mean(ane[mask]) - np.mean(ane[t < BASELINE])) if mask.sum() > 0 and (t < BASELINE).sum() > 0 else 0
    return t, ane, delta

# Compute mean deltas
white_deltas = []
black_deltas = []
for r in range(N_REPS):
    _, _, wd = load_ane_stats(f"resnet_power_white_r{r}.csv")
    _, _, bd = load_ane_stats(f"resnet_power_black_r{r}.csv")
    if wd is not None: white_deltas.append(wd)
    if bd is not None: black_deltas.append(bd)

w_mean = np.mean(white_deltas) if white_deltas else 0
b_mean = np.mean(black_deltas) if black_deltas else 0
closest_w = int(np.argmin([abs(d - w_mean) for d in white_deltas]))
closest_b = int(np.argmin([abs(d - b_mean) for d in black_deltas]))

t_w, ane_w, _ = load_ane_stats(f"resnet_power_white_r{closest_w}.csv")
t_b, ane_b, _ = load_ane_stats(f"resnet_power_black_r{closest_b}.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle('Apple Neural Engine (ANE) Power Signature: Black vs White Input',
             fontsize=16, fontweight='bold', y=0.95)

for ax, t, ane, title, color, label in [
    (ax1, t_w, ane_w, 'White Image (All 1s — Dense MAC Operations)', '#D62728', f'White (Ones)'),
    (ax2, t_b, ane_b, 'Black Image (All 0s — Zero-skipping Optimization)', '#1F77B4', f'Black (Zeros)'),
]:
    if t is None or ane is None:
        continue
    ax.plot(t, ane, color=LINE_COLOR, linewidth=1.5)
    ax.fill_between(t, ane, 0, color=FILL_COLOR, alpha=0.8)
    ax.axvline(x=BASELINE, color=LINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=BASELINE+INFERENCE, color=LINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.set_title(title, fontsize=12, pad=10, loc='left')
    ax.set_ylabel('Power (W)', fontsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    mask = (t >= BASELINE) & (t <= BASELINE + INFERENCE)
    if mask.sum() > 0:
        avg = np.mean(ane[mask])
        peak = np.max(ane[mask])
        ax.text(0.02, 0.85, f'Avg Inference Power: {avg:.4f} W',
                transform=ax.transAxes, fontsize=11, fontweight='bold', color='#B22222')

ax2.set_xlabel('Time (Seconds)', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
ts = time.strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(FIGURES_DIR, f'ane_power_comparison_{ts}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved -> {out_path}")
plt.close()
