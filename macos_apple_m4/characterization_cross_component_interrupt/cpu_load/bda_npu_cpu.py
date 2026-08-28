#!/usr/bin/env python3
"""
BDA: NPU power impact on CPU (P-core / E-core).
Measures how NPU full load affects P-core and E-core power consumption.

Experiments:
  E1: P-core normal + NPU full  (CPU on P-cores first, then NPU joins)
  E2: P-core full   + NPU full
  E3: E-core normal + NPU full
  E4: E-core full   + NPU full
"""
import subprocess, time, os, signal, sys
from datetime import datetime

MONITOR_BIN = "external/experiments_organized/02_npu_cpu_impact/cross_experiment/npu_cpu/neo_monitor_v2"
RESULTS_DIR = "external/experiments_organized/02_npu_cpu_impact/results_v2"
PYTHON_BIN = "/usr/bin/python3"
WORKER_DIR = "external/experiments_organized/02_npu_cpu_impact/cross_experiment/npu_cpu"

CPU_WORKER = f"{WORKER_DIR}/cpu_worker.py"
NPU_WORKER = f"{WORKER_DIR}/npu_worker_v3.py"

PHASE_BEFORE = 15
PHASE_DURING = 20
PHASE_AFTER  = 15

EXPERIMENTS = [
    ("E1_Pcore_normal_NPU_full", "p", "normal", "full"),
    ("E2_Pcore_full_NPU_full",   "p", "full",   "full"),
    ("E3_Ecore_normal_NPU_full", "e", "normal", "full"),
    ("E4_Ecore_full_NPU_full",   "e", "full",   "full"),
]

def cleanup():
    for name in ['cpu_worker', 'npu_worker', 'ANEPowerMonitor']:
        try:
            subprocess.run(['pkill', '-9', '-f', name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except:
            pass
    time.sleep(1)

def run_experiment(name, cpu_core, cpu_level, npu_level):
    print(f"\n  [cleanup] pre-run cleanup...")
    cleanup()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{name}_{ts}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)

    total_cpu_dur = PHASE_BEFORE + PHASE_DURING + PHASE_AFTER + 5
    npu_dur = PHASE_DURING + 5

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  CPU ({cpu_core}-core, {cpu_level}) runs {total_cpu_dur}s continuously")
    print(f"  NPU ({npu_level}) joins at t={PHASE_BEFORE+2}s for {npu_dur}s")
    print(f"{'='*65}")

    cpu_proc = npu_proc = mon_proc = None

    try:
        # 1. Monitor
        mon_proc = subprocess.Popen([MONITOR_BIN, csv_name], cwd=RESULTS_DIR,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # 2. CPU worker
        cpu_cmd = [PYTHON_BIN, CPU_WORKER, cpu_core, cpu_level, "--duration", str(total_cpu_dur)]
        print(f"  [1] CPU: {cpu_core}-core {cpu_level}")
        cpu_proc = subprocess.Popen(cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(PHASE_BEFORE)

        # 3. NPU joins
        npu_cmd = [PYTHON_BIN, NPU_WORKER, npu_level, "--duration", str(npu_dur)]
        print(f"  [2] +NPU: {npu_level}")
        npu_proc = subprocess.Popen(npu_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(PHASE_DURING)

        # 4. NPU exits
        print(f"  [3] -NPU")
        npu_proc.terminate()
        try: npu_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: npu_proc.kill()
        npu_proc = None
        time.sleep(PHASE_AFTER)

        # 5. CPU done
        print(f"  [4] Stop CPU")
        cpu_proc.terminate()
        try: cpu_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: cpu_proc.kill()
        cpu_proc = None
        time.sleep(3)

        # 6. Monitor done
        mon_proc.send_signal(signal.SIGTERM)
        try: mon_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: mon_proc.kill()
        mon_proc = None

    finally:
        for p in [cpu_proc, npu_proc, mon_proc]:
            if p and p.poll() is None:
                p.kill()

    if os.path.exists(csv_path):
        print(f"  OK: {csv_name} ({os.path.getsize(csv_path)/1024:.0f} KB)")
        return csv_name
    else:
        print(f"  FAIL: CSV not created")
        return None

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Global pre-cleanup...")
    cleanup()

    for name, core, cpu_lvl, npu_lvl in EXPERIMENTS:
        run_experiment(name, core, cpu_lvl, npu_lvl)
        cleanup()
        time.sleep(3)

    print("\nAll experiments done.")
