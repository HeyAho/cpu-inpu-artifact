#!/usr/bin/env python3
"""Generate styled ANE power comparison plots (matching original style)."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Style config (matching original figures)
LINE_COLOR = '#324F7D'
FILL_COLOR = '#C5D0E0'
BACKGROUND_COLOR = '#FFFFFF'

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR
plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_power_data(csv_path, power_column="ANE"):
    """Load CSV and extract time + specific power column."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None, None

    time_series = df.iloc[:, 0] - df.iloc[0, 0]

    if power_column in df.columns:
        power = df[power_column]
    else:
        # Fall back to sum of all channels
        power = df[df.columns[1:]].sum(axis=1)

    return time_series, power


def format_subplot(ax, time_data, power, title, inf_start, inf_end):
    """Style a subplot matching the original figure style."""
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
        ax.text(0.02, 0.85, f'Avg: {avg_pwr:.4f} W  Peak: {peak_pwr:.4f} W',
                transform=ax.transAxes, fontsize=11, fontweight='bold', color='#B22222')


def plot_power_comparison(black_csv, white_csv, power_column="ANE"):
    """Generate two-panel power comparison plot."""
    t_black, pwr_black = load_power_data(black_csv, power_column)
    t_white, pwr_white = load_power_data(white_csv, power_column)

    if t_black is None or t_white is None:
        return

    # Determine inference window from data characteristics
    inf_start = 5.0  # baseline is 5s
    inf_end = 35.0   # inference is 30s (5+30)
    cooldown_start = 35.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('Apple Neural Engine (ANE) Power Signature: Black vs White Input',
                 fontsize=16, fontweight='bold', y=0.95)

    format_subplot(ax1, t_white, pwr_white,
                   "White Image (All 1s — Dense MAC Operations)",
                   inf_start, inf_end)

    format_subplot(ax2, t_black, pwr_black,
                   "Black Image (All 0s — Zero-skipping Optimization)",
                   inf_start, inf_end)

    ax2.set_xlabel('Time (Seconds)', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    current_time = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f'ane_power_comparison_{current_time}.png'
    output_path = os.path.join(FIGURES_DIR, output_filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {output_path}")
    plt.close()

    return output_path


def plot_full_system_comparison(black_csv, white_csv):
    """Generate comparison plot showing total system power."""
    t_black, pwr_black = load_power_data(black_csv, power_column=None)
    t_white, pwr_white = load_power_data(white_csv, power_column=None)

    if t_black is None or t_white is None:
        return

    inf_start, inf_end = 5.0, 35.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('Total System Power: Black vs White Input (ResNet-50 on ANE)',
                 fontsize=16, fontweight='bold', y=0.95)

    format_subplot(ax1, t_white, pwr_white,
                   "White Image (All 1s) — Total System Power",
                   inf_start, inf_end)

    format_subplot(ax2, t_black, pwr_black,
                   "Black Image (All 0s) — Total System Power",
                   inf_start, inf_end)

    ax2.set_xlabel('Time (Seconds)', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    current_time = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(FIGURES_DIR, f'ane_power_comparison_total_{current_time}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {output_path}")
    plt.close()

    return output_path


def plot_side_by_side(black_csv, white_csv, power_column="ANE"):
    """Side-by-side comparison of black vs white ANE power."""
    t_black, pwr_black = load_power_data(black_csv, power_column)
    t_white, pwr_white = load_power_data(white_csv, power_column)

    if t_black is None or t_white is None:
        return

    inf_start, inf_end = 5.0, 35.0

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('ANE Power: White vs Black Input (ResNet-50)',
                 fontsize=16, fontweight='bold', y=0.98)

    ax.plot(t_white, pwr_white, color='#D62728', linewidth=1.0, alpha=0.7, label='White (Ones)')
    ax.plot(t_black, pwr_black, color='#1F77B4', linewidth=1.0, alpha=0.7, label='Black (Zeros)')

    ax.axvline(x=inf_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=inf_end, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(inf_start + 0.5, ax.get_ylim()[1]*0.95, 'Inference Start', fontsize=9, color='gray')
    ax.text(inf_end + 0.5, ax.get_ylim()[1]*0.95, 'Inference End', fontsize=9, color='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (Seconds)', fontsize=11)
    ax.set_ylabel('ANE Power (W)', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(fontsize=12)

    # Compute stats
    mask_b = (t_black >= inf_start) & (t_black <= inf_end)
    mask_w = (t_white >= inf_start) & (t_white <= inf_end)
    avg_b = pwr_black[mask_b].mean() if mask_b.any() else 0
    avg_w = pwr_white[mask_w].mean() if mask_w.any() else 0
    ratio = avg_w / avg_b if avg_b > 0 else 0

    stats_text = (
        f"White avg: {avg_w:.4f} W\n"
        f"Black avg: {avg_b:.4f} W\n"
        f"Ratio (W/B): {ratio:.2f}x"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    current_time = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(FIGURES_DIR, f'ane_power_side_by_side_{current_time}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    black_csv = os.path.join(OUTPUT_DIR, "resnet_power_black.csv")
    white_csv = os.path.join(OUTPUT_DIR, "resnet_power_white.csv")

    print("Generating ANE-only comparison (original style)...")
    plot_power_comparison(black_csv, white_csv, power_column="ANE")

    print("\nGenerating total system power comparison...")
    plot_full_system_comparison(black_csv, white_csv)

    print("\nGenerating side-by-side overlay...")
    plot_side_by_side(black_csv, white_csv, power_column="ANE")

    print(f"\nAll plots saved to: {FIGURES_DIR}/")
