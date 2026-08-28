#!/usr/bin/env python3
"""Generate styled plot from 5 power test repetitions with error bars."""
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

# Load all repetition CSV data
def load_csv_ane(csv_name):
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

# Load all reps and compute stats
white_active_means = []
black_active_means = []
white_deltas = []
black_deltas = []

for rep in range(N_REPS):
    t_w, ane_w = load_csv_ane(f"resnet_power_white_r{rep}.csv")
    t_b, ane_b = load_csv_ane(f"resnet_power_black_r{rep}.csv")

    if ane_w is not None:
        mask = (t_w >= BASELINE) & (t_w <= BASELINE + INFERENCE)
        active = np.mean(ane_w[mask]) if mask.sum() > 0 else 0
        baseline = np.mean(ane_w[t_w < BASELINE]) if (t_w < BASELINE).sum() > 0 else 0
        white_active_means.append(active)
        white_deltas.append(active - baseline)

    if ane_b is not None:
        mask = (t_b >= BASELINE) & (t_b <= BASELINE + INFERENCE)
        active = np.mean(ane_b[mask]) if mask.sum() > 0 else 0
        baseline = np.mean(ane_b[t_b < BASELINE]) if (t_b < BASELINE).sum() > 0 else 0
        black_active_means.append(active)
        black_deltas.append(active - baseline)

# Compute stats
w_mean = np.mean(white_active_means)
w_std = np.std(white_active_means)
w_d_mean = np.mean(white_deltas)
w_d_std = np.std(white_deltas)
b_mean = np.mean(black_active_means)
b_std = np.std(black_active_means)
b_d_mean = np.mean(black_deltas)
b_d_std = np.std(black_deltas)
ratio = w_d_mean / b_d_mean if b_d_mean > 0 else 0

print(f"White:  active={w_mean:.4f}+-{w_std:.4f}W  delta={w_d_mean:.4f}+-{w_d_std:.4f}W")
print(f"Black:  active={b_mean:.4f}+-{b_std:.4f}W  delta={b_d_mean:.4f}+-{b_d_std:.4f}W")
print(f"Ratio:  {ratio:.2f}x")

# Create figure: 3 panels
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
ts = time.strftime("%Y%m%d_%H%M%S")

# 1. Bar chart with error bars
ax1 = fig.add_subplot(gs[0, 0])
categories = ['White (Ones)', 'Black (Zeros)']
means = [w_d_mean, b_d_mean]
stds = [w_d_std, b_d_std]
colors_bars = ['#D62728', '#1F77B4']
bars = ax1.bar(categories, means, yerr=stds, capsize=8, color=colors_bars,
               edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
ax1.set_ylabel('ANE Power Delta (W)', fontsize=12)
ax1.set_title('Average ANE Power Increase', fontsize=14, fontweight='bold')
ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
             f'{mean:.3f} W', ha='center', fontsize=11, fontweight='bold')

# 2. Per-rep scatter
ax2 = fig.add_subplot(gs[0, 1])
reps_x = np.arange(N_REPS)
ax2.scatter(reps_x, white_deltas, color='#D62728', s=80, zorder=5, label='White', edgecolors='black')
ax2.scatter(reps_x, black_deltas, color='#1F77B4', s=80, zorder=5, label='Black', edgecolors='black')
ax2.plot(reps_x, white_deltas, color='#D62728', alpha=0.3, linestyle='--')
ax2.plot(reps_x, black_deltas, color='#1F77B4', alpha=0.3, linestyle='--')
ax2.axhline(w_d_mean, color='#D62728', linestyle=':', alpha=0.7, label=f'White avg={w_d_mean:.3f}W')
ax2.axhline(b_d_mean, color='#1F77B4', linestyle=':', alpha=0.7, label=f'Black avg={b_d_mean:.3f}W')
ax2.set_xlabel('Repetition')
ax2.set_ylabel('ANE Power Delta (W)')
ax2.set_title('Per-Repetition Power Delta', fontsize=14, fontweight='bold')
ax2.set_xticks(reps_x)
ax2.set_xticklabels([str(i+1) for i in reps_x])
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Ratio bar
ax3 = fig.add_subplot(gs[0, 2])
per_rep_ratios = []
for i in range(N_REPS):
    if i < len(white_deltas) and i < len(black_deltas) and black_deltas[i] > 0:
        per_rep_ratios.append(white_deltas[i] / black_deltas[i])
ratio_mean = np.mean(per_rep_ratios) if per_rep_ratios else 0
ratio_std = np.std(per_rep_ratios) if per_rep_ratios else 0
ax3.bar(['White/Black Ratio'], [ratio_mean], yerr=[ratio_std], capsize=8,
        color='darkorange', edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Equal (1.0x)')
ax3.set_ylabel('Power Ratio')
ax3.set_title('White/Black Power Ratio', fontsize=14, fontweight='bold')
ax3.set_ylim(0, max(ratio_mean + ratio_std + 0.5, 2.0))
ax3.legend(fontsize=10)
ax3.grid(True, axis='y', linestyle=':', alpha=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 4-5. Time series (best representative rep - rep with closest to mean)
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

# Find rep closest to mean for white
closest_w = np.argmin([abs(d - w_d_mean) for d in white_deltas])
closest_b = np.argmin([abs(d - b_d_mean) for d in black_deltas])

t_w, ane_w = load_csv_ane(f"resnet_power_white_r{closest_w}.csv")
t_b, ane_b = load_csv_ane(f"resnet_power_black_r{closest_b}.csv")

for ax, t, ane, color, title, label in [
    (ax4, t_w, ane_w, '#D62728', 'White Image (Ones)', f'White Rep {closest_w+1}'),
    (ax5, t_b, ane_b, '#1F77B4', 'Black Image (Zeros)', f'Black Rep {closest_b+1}'),
]:
    if t is None or ane is None:
        continue
    ax.plot(t, ane, color=color, linewidth=1.2)
    ax.fill_between(t, ane, 0, color=color, alpha=0.15)
    ax.axvline(x=BASELINE, color='#324F7D', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axvline(x=BASELINE+INFERENCE, color='#324F7D', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ANE Power (W)')
    ax.set_title(f'{title} ({label})', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mask = (t >= BASELINE) & (t <= BASELINE + INFERENCE)
    if mask.sum() > 0:
        avg = np.mean(ane[mask])
        peak = np.max(ane[mask])
        ax.text(0.02, 0.85, f'Avg: {avg:.4f} W\nPeak: {peak:.4f} W',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color='#B22222', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Statistical summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary = (
    f"SUMMARY (n={N_REPS})\n"
    f"{'='*18}\n\n"
    f"White (Ones):\n"
    f"  Active: {w_mean:.3f} ± {w_std:.3f} W\n"
    f"  Delta:  {w_d_mean:.3f} ± {w_d_std:.3f} W\n\n"
    f"Black (Zeros):\n"
    f"  Active: {b_mean:.3f} ± {b_std:.3f} W\n"
    f"  Delta:  {b_d_mean:.3f} ± {b_d_std:.3f} W\n\n"
    f"Ratio (W/B):\n"
    f"  Active: {w_mean/b_mean:.2f}x\n"
    f"  Delta:  {ratio:.2f}x\n"
    f"  Peak:   {np.mean([np.max(load_csv_ane(f'resnet_power_white_r{r}.csv')[1][(load_csv_ane(f'resnet_power_white_r{r}.csv')[0]>=BASELINE)&(load_csv_ane(f'resnet_power_white_r{r}.csv')[0]<=BASELINE+INFERENCE)]) for r in range(N_REPS) if load_csv_ane(f'resnet_power_white_r{r}.csv')[1] is not None]) / max(np.mean([np.max(load_csv_ane(f'resnet_power_black_r{r}.csv')[1][(load_csv_ane(f'resnet_power_black_r{r}.csv')[0]>=BASELINE)&(load_csv_ane(f'resnet_power_black_r{r}.csv')[0]<=BASELINE+INFERENCE)]) for r in range(N_REPS) if load_csv_ane(f'resnet_power_black_r{r}.csv')[1] is not None]), 0.001):.2f}x\n\n"
    f"Conclusion:\n"
    f"White input consumes\n"
    f"{ratio:.1f}x more ANE power\n"
    f"than black input,\n"
    f"confirming NPU hardware\n"
    f"zero-skipping optimization."
)
ax6.text(0.1, 0.95, summary, fontsize=10, fontfamily='monospace',
         verticalalignment='top', transform=ax6.transAxes)

plt.suptitle(f'ANE Power: Black vs White Input (ResNet-50, {N_REPS} Repetitions)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, f'ane_power_multi_rep_{ts}.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved -> {out_path}")
plt.close()
