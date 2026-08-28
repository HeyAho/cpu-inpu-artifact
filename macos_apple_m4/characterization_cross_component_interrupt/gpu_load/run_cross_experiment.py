#!/usr/bin/env python3
"""
GPU-NPU 交叉功耗实验 v2

实验矩阵:
  Cross (2x2):
    1. GPU_normal (N=128)  + NPU_full   (WideResNet101_2 B16 S512)
    2. GPU_normal (N=128)  + NPU_normal (ResNet50 B4 S16)
    3. GPU_full   (N=2048) + NPU_full   (WideResNet101_2 B16 S512)
    4. GPU_full   (N=2048) + NPU_normal (ResNet50 B4 S16)

  Solo (baseline):
    5. GPU_normal alone
    6. GPU_full alone
    7. NPU_full alone
    8. NPU_normal alone

Timeline per run:
  0-2s:   baseline (idle)
  2-7s:   primary task stabilize
  7-32s:  both tasks (cross) or primary continues (solo)
  32-37s: cooldown
"""
import subprocess, time, os, signal, sys, json
from datetime import datetime

MONITOR_BIN = "external/experiments_organized/03_npu_gpu_impact/cross_experiment/neo_monitor_v2"
RESULTS_DIR = "external/experiments_organized/03_npu_gpu_impact/results_v2"
PYTHON_BIN = "/usr/bin/python3"
WORKER_DIR = "external/experiments_organized/03_npu_gpu_impact/cross_experiment"

RUN_DURATION = 25
STABILIZE_TIME = 5

def run_experiment(name, cmds, is_cross):
    """Run one experiment. cmds: list of (worker, level) tuples"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{name}_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # 1. Start monitor
    print(f"  [1] Monitor start...")
    monitor = subprocess.Popen([MONITOR_BIN, csv_name], cwd=RESULTS_DIR,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # 2. Start primary workload
    primary_cmd = [PYTHON_BIN, cmds[0][0], cmds[0][1],
                   "--duration", str(STABILIZE_TIME + RUN_DURATION + 5)]
    print(f"  [2] Primary: {cmds[0][0].split('/')[-1]} {cmds[0][1]}")
    primary = subprocess.Popen(primary_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(STABILIZE_TIME)

    # 3. Start secondary (cross only)
    secondary = None
    if is_cross and len(cmds) > 1:
        secondary_cmd = [PYTHON_BIN, cmds[1][0], cmds[1][1],
                         "--duration", str(RUN_DURATION + 5)]
        print(f"  [3] Secondary: {cmds[1][0].split('/')[-1]} {cmds[1][1]}")
        secondary = subprocess.Popen(secondary_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # 4. Run
    print(f"  [4] Running {RUN_DURATION}s...")
    time.sleep(RUN_DURATION)

    # 5. Stop secondary first
    if secondary:
        print(f"  [5] Stop secondary...")
        secondary.terminate()
        try: secondary.wait(timeout=5)
        except subprocess.TimeoutExpired: secondary.kill()
        time.sleep(2)

    # 6. Stop primary
    print(f"  [6] Stop primary...")
    primary.terminate()
    try: primary.wait(timeout=5)
    except subprocess.TimeoutExpired: primary.kill()
    time.sleep(2)

    # 7. Stop monitor
    print(f"  [7] Stop monitor...")
    monitor.send_signal(signal.SIGTERM)
    try: monitor.wait(timeout=5)
    except subprocess.TimeoutExpired: monitor.kill()

    if os.path.exists(csv_path):
        print(f"  OK: {csv_name} ({os.path.getsize(csv_path)/1024:.0f} KB)")
        return csv_name
    else:
        print(f"  FAIL: CSV not generated")
        return None

GPU_WORKER = f"{WORKER_DIR}/gpu_worker_v2.py"
NPU_WORKER = f"{WORKER_DIR}/npu_worker_v3.py"

EXPERIMENTS = [
    # Cross: GPU + NPU simultaneously
    ("X1_gpu_normal_npu_full",   [(GPU_WORKER, "normal"), (NPU_WORKER, "full")],   True),
    ("X2_gpu_normal_npu_normal", [(GPU_WORKER, "normal"), (NPU_WORKER, "normal")], True),
    ("X3_gpu_full_npu_full",     [(GPU_WORKER, "full"),   (NPU_WORKER, "full")],   True),
    ("X4_gpu_full_npu_normal",   [(GPU_WORKER, "full"),   (NPU_WORKER, "normal")], True),
    # Solo baselines
    ("S1_gpu_normal_solo", [(GPU_WORKER, "normal")], False),
    ("S2_gpu_full_solo",   [(GPU_WORKER, "full")],   False),
    ("S3_npu_full_solo",   [(NPU_WORKER, "full")],   False),
    ("S4_npu_normal_solo", [(NPU_WORKER, "normal")], False),
]

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    for name, cmds, is_cross in EXPERIMENTS:
        csv_name = run_experiment(name, cmds, is_cross)
        all_results.append({"name": name, "csv": csv_name, "cross": is_cross})

    # Save index
    index_path = os.path.join(RESULTS_DIR, "experiment_index.json")
    with open(index_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for r in all_results:
        status = "OK" if r["csv"] else "FAIL"
        print(f"  [{status}] {r['name']} -> {r['csv']}")
    print(f"\n  Index: {index_path}")
