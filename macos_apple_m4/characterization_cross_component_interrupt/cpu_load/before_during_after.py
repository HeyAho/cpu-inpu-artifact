#!/usr/bin/env python3
"""
Before-During-After 功耗实验 (v2: 实验间自动清理)

测量一个部件 (primary) 在另一个部件 (secondary) 加入前后的功耗变化。
"""
import subprocess, time, os, signal, sys
from datetime import datetime

MONITOR_BIN = "external/ANEPowerMonitor"
RESULTS_DIR = "results/before_during_after"
PYTHON_BIN = "/usr/bin/python3"
WORKER_DIR = "."

GPU_WORKER = f"{WORKER_DIR}/gpu_worker_v2.py"
NPU_WORKER = f"{WORKER_DIR}/npu_worker_v3.py"

PHASE_BEFORE = 15
PHASE_DURING = 20
PHASE_AFTER  = 15

EXPERIMENTS = [
    ("E1_GPU_normal_add_NPU_full", GPU_WORKER, "normal", NPU_WORKER, "full", "GPU Energy"),
    ("E2_GPU_full_add_NPU_full",   GPU_WORKER, "full",   NPU_WORKER, "full", "GPU Energy"),
    ("E3_NPU_normal_add_GPU_full", NPU_WORKER, "normal", GPU_WORKER, "full", "ANE"),
    ("E4_NPU_full_add_GPU_full",   NPU_WORKER, "full",   GPU_WORKER, "full", "ANE"),
]

def cleanup():
    """强制清理所有可能的残留进程"""
    for name in ['test.py', 'gpu_worker', 'npu_worker', 'ANEPowerMonitor']:
        try:
            subprocess.run(['pkill', '-9', '-f', name], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except:
            pass
    time.sleep(1)

def verify_clean():
    """确认没有残留进程"""
    result = subprocess.run(
        "ps aux | grep -E 'test\.py|gpu_worker|npu_worker|ANEPowerMonitor' | grep -v grep | grep -v vscode",
        shell=True, capture_output=True, text=True, timeout=5)
    lingering = result.stdout.strip()
    if lingering:
        print(f"  ⚠ 残留进程:\n{lingering}")
        cleanup()
        return False
    return True

def run_bda_experiment(name, primary_worker, primary_level, secondary_worker, secondary_level):
    # 实验前清理检查
    print(f"\n  [cleanup] 实验前清理...")
    cleanup()
    if not verify_clean():
        print(f"  [cleanup] 强制清理后仍有残留，继续...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{name}_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)

    total_primary_duration = PHASE_BEFORE + PHASE_DURING + PHASE_AFTER + 5
    secondary_duration = PHASE_DURING + 5

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  Primary runs continuously for {total_primary_duration}s")
    print(f"  Secondary joins at t={PHASE_BEFORE+2}s for {PHASE_DURING}s")
    print(f"{'='*65}")

    primary = None
    secondary = None
    monitor = None

    try:
        # 1. Start monitor
        monitor = subprocess.Popen([MONITOR_BIN, csv_name], cwd=RESULTS_DIR,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # 2. Start primary
        primary_cmd = [PYTHON_BIN, primary_worker, primary_level, "--duration", str(total_primary_duration)]
        print(f"  [1] Primary: {primary_worker.split('/')[-1]} {primary_level}")
        primary = subprocess.Popen(primary_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(PHASE_BEFORE)

        # 3. Start secondary
        secondary_cmd = [PYTHON_BIN, secondary_worker, secondary_level, "--duration", str(secondary_duration)]
        print(f"  [2] +Secondary: {secondary_worker.split('/')[-1]} {secondary_level}")
        secondary = subprocess.Popen(secondary_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(PHASE_DURING)

        # 4. Stop secondary
        print(f"  [3] -Secondary")
        secondary.terminate()
        try: secondary.wait(timeout=5)
        except subprocess.TimeoutExpired: secondary.kill()
        secondary = None
        time.sleep(PHASE_AFTER)

        # 5. Stop primary
        print(f"  [4] Stop primary")
        primary.terminate()
        try: primary.wait(timeout=5)
        except subprocess.TimeoutExpired: primary.kill()
        primary = None
        time.sleep(3)

        # 6. Stop monitor
        monitor.send_signal(signal.SIGTERM)
        try: monitor.wait(timeout=5)
        except subprocess.TimeoutExpired: monitor.kill()
        monitor = None

    finally:
        # 确保所有进程被终止
        for proc, label in [(primary, "primary"), (secondary, "secondary"), (monitor, "monitor")]:
            if proc is not None and proc.poll() is None:
                print(f"  [cleanup] 终止残留 {label}...")
                proc.kill()
                try: proc.wait(timeout=3)
                except: pass

    if os.path.exists(csv_path):
        size_kb = os.path.getsize(csv_path) / 1024
        print(f"  OK: {csv_name} ({size_kb:.0f} KB)")
        return csv_name
    else:
        print(f"  FAIL: CSV not created")
        return None

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 启动前全局清理
    print("实验前全局清理...")
    cleanup()

    for name, pw, pl, sw, sl, mc in EXPERIMENTS:
        run_bda_experiment(name, pw, pl, sw, sl)
        # 实验后额外清理
        print(f"  [cleanup] 实验 {name} 完成，清理残留...")
        cleanup()
        time.sleep(2)  # 间隔冷却

    print("\n全部实验完成。")
