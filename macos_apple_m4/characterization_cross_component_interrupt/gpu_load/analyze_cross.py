#!/usr/bin/env python3
"""
分析 GPU-NPU 交叉功耗实验 v2

对比交叉运行与 solo 基线，计算功耗影响百分比
"""
import os, sys, csv, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("INPU_CROSS_RESULTS_DIR", str(SCRIPT_DIR / "results"))

def load_csv(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            try: rows.append([float(v) for v in row])
            except (ValueError, IndexError): continue
    return headers, rows

def find_cols(headers):
    """Find ANE and GPU Energy column indices"""
    cols = {}
    for i, h in enumerate(headers):
        if h == "ANE": cols["ane"] = i
        elif h == "GPU Energy": cols["gpu_e"] = i
        elif h == "PCPU": cols["pcpu"] = i
    return cols

def phase_avg(rows, times_from_start, cols, t_start, t_end):
    """Average values in time window [t_start, t_end)"""
    phase_rows = [r for r, t in zip(rows, times_from_start) if t_start <= t < t_end]
    if not phase_rows: return {}
    n = len(phase_rows)
    result = {}
    for key, ci in cols.items():
        vals = [r[ci] for r in phase_rows]
        result[key] = sum(vals) / n
        result[f"{key}_active"] = 100.0 * sum(1 for v in vals if v > 0.01) / n
    return result

def analyze_csv(csv_path):
    headers, rows = load_csv(csv_path)
    if not rows: return None
    cols = find_cols(headers)
    t0 = rows[0][0]
    times = [r[0] - t0 for r in rows]

    return {
        "baseline":  phase_avg(rows, times, cols, 0, 2),
        "primary":   phase_avg(rows, times, cols, 2, 7),
        "both":      phase_avg(rows, times, cols, 7, 32),
        "cooldown":  phase_avg(rows, times, cols, 34, 38),
    }

def get_val(results, phase, metric):
    return results.get(phase, {}).get(metric, 0)

def main():
    if not os.path.isdir(RESULTS_DIR):
        print("Results dir not found"); sys.exit(1)

    # Load all CSVs
    csv_files = sorted([f for f in os.listdir(RESULTS_DIR)
                        if f.endswith(".csv") and (f.startswith("X") or f.startswith("S"))])
    data = {}
    for cf in csv_files:
        name = cf.split("_", 1)[1].rsplit("_", 1)[0]  # e.g. "gpu_normal_npu_full"
        prefix = cf[0]  # X or S
        key = f"{prefix}_{name.replace('_solo','')}"
        r = analyze_csv(os.path.join(RESULTS_DIR, cf))
        if r:
            data[key] = r

    # Solo baselines (use "primary" phase since there's no secondary task)
    solo = {}
    for skey in ["S_gpu_normal", "S_gpu_full", "S_npu_full", "S_npu_normal"]:
        if skey in data:
            solo[skey] = data[skey]["primary"]

    print("=" * 90)
    print("  GPU-NPU Cross Power Experiment v2 Results")
    print("  GPU Energy unit: raw (divide by 1e6 for Watts)")
    print("=" * 90)

    print("\n--- Solo Baselines ---")
    print(f"{'Condition':<25} {'GPU Energy':>14} {'ANE':>10} {'ANE act%':>10}")
    print("-" * 60)
    for skey in ["S_gpu_normal", "S_gpu_full", "S_npu_full", "S_npu_normal"]:
        if skey in solo:
            s = solo[skey]
            print(f"{skey:<25} {s.get('gpu_e',0):>12.0f} W {s.get('ane',0):>10.2f} {s.get('ane_active',0):>8.1f}%")

    print("\n--- Cross Experiments: GPU Power Impact ---")
    print(f"{'Experiment':<30} {'GPU solo(W)':>12} {'GPU cross(W)':>13} {'Delta':>8} {'Delta%':>8}")
    print("-" * 75)

    cross_map = {
        "X_gpu_normal_npu_full":   "S_gpu_normal",
        "X_gpu_normal_npu_normal": "S_gpu_normal",
        "X_gpu_full_npu_full":     "S_gpu_full",
        "X_gpu_full_npu_normal":   "S_gpu_full",
    }
    for xkey, skey in cross_map.items():
        if xkey in data and skey in solo:
            x_both = data[xkey]["both"]
            s = solo[skey]
            gpu_solo = s.get("gpu_e", 0)
            gpu_cross = x_both.get("gpu_e", 0)
            delta = gpu_cross - gpu_solo
            delta_pct = 100 * delta / gpu_solo if gpu_solo else 0
            print(f"{xkey:<30} {gpu_solo:>10.0f} W {gpu_cross:>11.0f} W {delta:>+8.0f} {delta_pct:>+7.1f}%")

    print("\n--- Cross Experiments: NPU (ANE) Power Impact ---")
    print(f"{'Experiment':<30} {'ANE solo':>10} {'ANE cross':>11} {'Delta':>8} {'Delta%':>8}")
    print("-" * 70)

    cross_map_ane = {
        "X_gpu_normal_npu_full":   "S_npu_full",
        "X_gpu_normal_npu_normal": "S_npu_normal",
        "X_gpu_full_npu_full":     "S_npu_full",
        "X_gpu_full_npu_normal":   "S_npu_normal",
    }
    for xkey, skey in cross_map_ane.items():
        if xkey in data and skey in solo:
            x_both = data[xkey]["both"]
            s = solo[skey]
            ane_solo = s.get("ane", 0)
            ane_cross = x_both.get("ane", 0)
            delta = ane_cross - ane_solo
            delta_pct = 100 * delta / ane_solo if ane_solo else 0
            print(f"{xkey:<30} {ane_solo:>10.2f} {ane_cross:>11.2f} {delta:>+8.2f} {delta_pct:>+7.1f}%")

    print("\n--- Full Comparison Matrix ---")
    print(f"{'':<25} {'GPU normal':>14} {'GPU full':>14}")
    for npu_label, npu_skey in [("NPU normal (ANE solo)", "S_npu_normal"), ("NPU full (ANE solo)", "S_npu_full")]:
        if npu_skey in solo:
            ane_solo = solo[npu_skey].get("ane", 0)
            gpu_normal_cross = data.get("X_gpu_normal_" + npu_skey.split("_",1)[1], {}).get("both", {}).get("ane", 0)
            gpu_full_cross = data.get("X_gpu_full_" + npu_skey.split("_",1)[1], {}).get("both", {}).get("ane", 0)
            print(f"{npu_label:<25} {ane_solo:>14.2f} {'':>14}")
            print(f"{'  + GPU normal':<25} {gpu_normal_cross:>14.2f}")
            print(f"{'  + GPU full':<25} {gpu_full_cross:>14.2f}")

    # Save detailed data
    out = {}
    for k, v in data.items():
        out[k] = {phase: {metric: val for metric, val in metrics.items()}
                  for phase, metrics in v.items()}
    with open(os.path.join(RESULTS_DIR, "analysis_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDetailed data: {RESULTS_DIR}/analysis_v2.json")

if __name__ == "__main__":
    main()
