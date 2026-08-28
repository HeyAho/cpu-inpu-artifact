#!/usr/bin/env python3
"""
ANE-aware BDA 分析: 检测 ANE 功耗起始点，只统计 NPU 实际工作后的数据，
排除模型加载阶段的 CPU 干扰。

v2: 修复 ANE onset 晚于 T_ADD 时的边界情况
    - DURING 窗口改为从 max(T_ADD, ane_onset) 开始
    - BEFORE 窗口在 ane_onset >= T_ADD 时正确留空
    - Summary 在 ANE-aware BEFORE 为空时回退到 fixed-window BEFORE
"""
import os, csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = os.environ.get("INPU_CROSS_RESULTS_DIR", str(SCRIPT_DIR / "results_bda"))
ANE_THRESHOLD = 0.5  # ANE > 0.5 认为 NPU 已开始工作

def load(path):
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [[float(v) for v in row] for row in reader if row]
    return headers, rows

def find_ane_onset(rows, ane_col):
    """返回 ANE 首次超过阈值的相对时间(秒)，若未找到返回 None"""
    t0 = rows[0][0]
    for r in rows:
        if r[ane_col] > ANE_THRESHOLD:
            return r[0] - t0
    return None

def get_col(headers):
    """返回 {简写: 列索引}"""
    cols = {}
    for i, h in enumerate(headers):
        if h == "ANE": cols["ane"] = i
        elif h == "GPU Energy": cols["gpu_e"] = i
        elif h == "PCPU": cols["pcpu"] = i
    return cols

def avg_active(rows, cols, t0, ts, te, active_col=None, threshold=0):
    """窗口 [ts, te) 内平均值，可选 active_col > threshold 过滤"""
    if ts >= te:
        return {"n_rows": 0, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    sub = []
    for r in rows:
        t = r[0] - t0
        if ts <= t < te:
            if active_col is None or r[active_col] > threshold:
                sub.append(r)
    if not sub: return {"n_rows": 0, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    n = len(sub)
    out = {"n_rows": n, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    for name, ci in cols.items():
        vals = [r[ci] for r in sub]
        out[name] = sum(vals) / n
    return out

def avg_all(rows, cols, t0, ts, te):
    """全量平均（不过滤）"""
    if ts >= te:
        return {"n_rows": 0, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    sub = [r for r in rows if ts <= r[0] - t0 < te]
    if not sub: return {"n_rows": 0, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    n = len(sub)
    out = {"n_rows": n, "window": f"[{ts:.1f}s, {te:.1f}s)"}
    for name, ci in cols.items():
        vals = [r[ci] for r in sub]
        out[name] = sum(vals) / n
    return out

def get_val(data, key, measure_col):
    """取 measure_col 对应的值，若数据为空返回 None"""
    if not data or data.get("n_rows", 0) == 0:
        return None
    lookup = "ane" if measure_col == "ANE" else "gpu_e"
    return data.get(lookup)

def fmt(val):
    """格式化数值或 None"""
    if val is None: return "N/A"
    return f"{val:.2f}"

def fmt0(val):
    """格式化整数或 None"""
    if val is None: return "N/A"
    return f"{val:.0f}"

# Find latest CSVs
csv_map = {}
for cf in os.listdir(RESULTS):
    if not cf.endswith(".csv"): continue
    prefix = cf.split("_2026")[0]
    if prefix not in csv_map or cf > csv_map[prefix]:
        csv_map[prefix] = cf

# ---- Experiment configurations ----
configs = [
    ("E1_GPU_normal_add_NPU_full",   "GPU Energy", "GPU",   "NPU full"),
    ("E2_GPU_full_add_NPU_full",     "GPU Energy", "GPU",   "NPU full"),
    ("E3_NPU_normal_add_GPU_full",   "ANE",        "NPU",   "GPU full"),
    ("E4_NPU_full_add_GPU_full",     "ANE",        "NPU",   "GPU full"),
]

T_ADD    = 17   # secondary added
T_REMOVE = 37   # secondary removed
T_END    = 52   # primary stopped

print()
print("=" * 110)
print("  ANE-Aware BDA Analysis v2: Edge-case fixed")
print("  DURING window starts at max(T_ADD, ANE_onset) to exclude model loading")
print("  Summary falls back to fixed-window BEFORE when ANE-aware BEFORE is empty")
print("=" * 110)

for prefix, measure_col, primary_type, secondary_type in configs:
    if prefix not in csv_map: continue
    path = os.path.join(RESULTS, csv_map[prefix])
    headers, rows = load(path)
    cols = get_col(headers)
    t0 = rows[0][0]

    ane_onset = find_ane_onset(rows, cols.get("ane", -1))
    is_npu_primary = (secondary_type == "GPU full")  # E3/E4: NPU is primary

    print(f"\n  {'='*100}")
    print(f"  {prefix}")
    print(f"  Primary: {primary_type} (continuous)  |  Secondary: {secondary_type} (toggle {T_ADD}s-{T_REMOVE}s)")
    if ane_onset:
        timing_note = ""
        if ane_onset > T_ADD:
            timing_note = f"  ⚠ ANE 晚于 T_ADD={T_ADD}s 启动，NPU 模型加载延迟 ≈ {ane_onset - T_ADD:.1f}s"
        elif ane_onset < 2:
            timing_note = "  ANE 在预热期 (<2s) 即启动"
        print(f"  ANE onset: t={ane_onset:.1f}s{('  ' + timing_note) if timing_note else ''}")
    else:
        print(f"  ANE onset: NEVER DETECTED (NPU may not use ANE)")
    print(f"  {'='*100}")

    # ===== Fixed time windows (original) =====
    before_fixed = avg_all(rows, cols, t0, 2, T_ADD)
    during_fixed = avg_all(rows, cols, t0, T_ADD, T_REMOVE)
    after_fixed  = avg_all(rows, cols, t0, T_REMOVE, T_END)

    # ===== ANE-aware windows (v2: edge-case fixed) =====
    if ane_onset:
        # BEFORE: from ANE onset to T_ADD (only meaningful if ane_onset < T_ADD)
        before_start = max(2, ane_onset)
        # DURING: from max(T_ADD, ane_onset) to T_REMOVE — key fix
        during_start = max(T_ADD, ane_onset)
        # AFTER: from T_REMOVE to T_END (only rows with ANE active)
        # Keep as-is, but note if ANE is off
    else:
        before_start = 2
        during_start = T_ADD

    before_active = avg_all(rows, cols, t0, before_start, T_ADD)
    during_active = avg_active(rows, cols, t0, during_start, T_REMOVE,
                                active_col=cols.get("ane"), threshold=ANE_THRESHOLD)
    after_active  = avg_active(rows, cols, t0, T_REMOVE, T_END,
                                active_col=cols.get("ane"), threshold=ANE_THRESHOLD)

    # ===== Print comparison =====
    print(f"\n  {'Method':<25} {'Phase':<18} {'n_rows':>7} {'window':>22} {measure_col:>14} {'GPU Energy':>14} {'ANE':>10}")
    print(f"  {'-'*105}")

    for label, data in [("BEFORE (fixed)", before_fixed), ("DURING (fixed)", during_fixed), ("AFTER (fixed)", after_fixed)]:
        w = data.get("window", "")
        print(f"  {'Fixed window':<25} {label:<18} {data.get('n_rows',0):>7} {w:>22} {data.get(measure_col.lower() if measure_col == 'ANE' else 'gpu_e',0):>14.2f} {data.get('gpu_e',0):>14.0f} {data.get('ane',0):>10.2f}")

    print(f"  {'-'*105}")

    for label, data in [("BEFORE (ANE-aware)", before_active), ("DURING (ANE-aware)", during_active), ("AFTER (ANE-aware)", after_active)]:
        w = data.get("window", "")
        print(f"  {'ANE-aware':<25} {label:<18} {data.get('n_rows',0):>7} {w:>22} {get_val(data, 'gpu_e',measure_col) or 0:>14.2f} {data.get('gpu_e',0):>14.0f} {data.get('ane',0):>10.2f}")

    # ===== Key comparison: ANE-aware deltas (v2: fallback) =====
    b_val = get_val(before_active, "gpu_e", measure_col)
    d_val = get_val(during_active, "gpu_e", measure_col)
    a_val = get_val(after_active, "gpu_e", measure_col)

    b_gpu = before_active.get("gpu_e")
    d_gpu = during_active.get("gpu_e")
    a_gpu = after_active.get("gpu_e")
    b_ane = before_active.get("ane")
    d_ane = during_active.get("ane")
    a_ane = after_active.get("ane")

    print(f"\n  --- ANE-Aware Impact Summary ---")

    # When ANE-aware BEFORE is empty (ANE onset after T_ADD), fall back to fixed-window BEFORE
    b_fallback = False
    if b_val is None and before_fixed.get("n_rows", 0) > 0:
        b_val = before_fixed.get(measure_col.lower() if measure_col == "ANE" else "gpu_e")
        b_gpu = before_fixed.get("gpu_e")
        b_ane = before_fixed.get("ane")
        b_fallback = True

    if b_val is not None and d_val is not None:
        add_delta = d_val - b_val
        add_pct = 100 * add_delta / b_val if b_val else 0
        tag = " (using fixed-window baseline)" if b_fallback else ""
        print(f"  Adding {secondary_type}:   {measure_col} {b_val:.2f} -> {d_val:.2f}  ({add_delta:+.2f}, {add_pct:+.1f}%){tag}")
    else:
        missing = "baseline" if b_val is None else "during"
        print(f"  Adding {secondary_type}:   INCOMPLETE — {missing} data unavailable")

    if d_val is not None and a_val is not None:
        rm_delta = a_val - d_val
        rm_pct = 100 * rm_delta / d_val if d_val else 0
        print(f"  Removing {secondary_type}: {measure_col} {d_val:.2f} -> {a_val:.2f}  ({rm_delta:+.2f}, {rm_pct:+.1f}%)")
    elif d_val is not None and a_val is None:
        print(f"  Removing {secondary_type}: ANE turned off — no active AFTER data to compare")
    else:
        print(f"  Removing {secondary_type}: INCOMPLETE — data unavailable")

    print(f"  Corresponding GPU Energy:  {fmt0(b_gpu):>4} -> {fmt0(d_gpu):>4} -> {fmt0(a_gpu):>4}")
    print(f"  Corresponding ANE:         {fmt(b_ane):>4} -> {fmt(d_ane):>4} -> {fmt(a_ane):>4}")

    # Extra insight for late-ANE-onset experiments
    if ane_onset and ane_onset > T_ADD:
        # Compare fixed BEFORE (GPU-only) vs ANE-aware DURING (GPU+NPU co-running)
        b_fixed_val = before_fixed.get(measure_col.lower() if measure_col == "ANE" else "gpu_e")
        if b_fixed_val and d_val:
            fixed_delta = d_val - b_fixed_val
            fixed_pct = 100 * fixed_delta / b_fixed_val if b_fixed_val else 0
            print(f"  [Fixed-ref] Adding (vs fixed BEFORE): {b_fixed_val:.2f} -> {d_val:.2f}  ({fixed_delta:+.2f}, {fixed_pct:+.1f}%)")
            print(f"  [Note] ANE-aware DURING window [{during_start:.1f}s, {T_REMOVE}s), "
                  f"effective co-run duration = {T_REMOVE - during_start:.1f}s")

print()
