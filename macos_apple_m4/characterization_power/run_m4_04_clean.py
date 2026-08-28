#!/usr/bin/env python3
"""M4 04: Black vs White ANE power — clean separate runs, B=1 model."""
import subprocess, time, os, signal, numpy as np
import coremltools as ct

MONITOR_BIN = "external/experiments_organized/02_npu_cpu_impact/cross_experiment/npu_cpu/neo_monitor_v2"
OUTPUT_DIR = "external/experiments_organized/04_npu_power_black_white/results_v3"
MODEL_PATH = "external/experiments_organized/04_npu_power_black_white/resnet_power_test/ResNet50_ANE.mlpackage"

# Archive copies can be run without the original machine-local paths.  Set
# INPU_MONITOR or INPU_OUTPUT_DIR to override these defaults.
_archive_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MONITOR_BIN = os.environ.get("INPU_MONITOR", MONITOR_BIN)
OUTPUT_DIR = os.environ.get("INPU_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "data"))
MODEL_PATH = os.environ.get(
    "INPU_RESNET50_MODEL",
    os.path.join(
        _archive_root,
        "models",
        "by_platform",
        "macos_apple_m4",
        "characterization_power",
        "ResNet50_ANE.mlpackage",
    ),
)

N_REPS = 30
INFERENCE_DURATION = 15
BASELINE_DURATION = 5
WARMUP_ITERS = 50
COOLDOWN = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


def kill_all():
    """Kill all residual processes."""
    for name in ['neo_monitor', 'monitor_v2', 'python3']:
        try:
            subprocess.run(['pkill', '-9', '-f', name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except:
            pass
    time.sleep(2)


def run_rep(input_data, csv_name, label):
    print("  [%s] Starting..." % label, flush=True)

    # Kill any leftovers from previous run
    kill_all()

    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Start monitor
    mon = subprocess.Popen([MONITOR_BIN, csv_name], cwd=OUTPUT_DIR,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    time.sleep(BASELINE_DURATION)

    # Run inference
    start = time.time()
    iters = 0
    try:
        while time.time() - start < INFERENCE_DURATION:
            model.predict({"x": input_data})
            iters += 1
    except Exception as e:
        print("  [%s] ERROR: %s" % (label, e), flush=True)

    elapsed = time.time() - start
    time.sleep(COOLDOWN)

    # Stop monitor
    mon.send_signal(signal.SIGTERM)
    try:
        mon.wait(timeout=5)
    except subprocess.TimeoutExpired:
        mon.kill()
    time.sleep(1)

    # Verify output
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 5000:
        print("  [%s] OK: %d iters in %.1fs (%.0f/s), CSV=%dKB" % (
            label, iters, elapsed, iters/elapsed if elapsed > 0 else 0,
            os.path.getsize(csv_path) // 1024), flush=True)
    else:
        size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
        print("  [%s] WARN: CSV=%d bytes" % (label, size), flush=True)

    # Final cleanup
    kill_all()


# ===== Load model and warmup =====
print("Loading ResNet-50 B=1 (M4, ANE)...", flush=True)
model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
input_name = model.input_description._fd_spec[0].name
print("Input: %s" % input_name, flush=True)

# Warmup with black input
print("Warmup (%d iters)..." % WARMUP_ITERS, flush=True)
for i in range(WARMUP_ITERS):
    model.predict({input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
print("Warmup done.", flush=True)

# ===== Black pixels =====
print("\n" + "=" * 60)
print("BLACK PIXELS (all zeros) — %d reps" % N_REPS)
print("=" * 60, flush=True)

black_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
for rep in range(N_REPS):
    run_rep(black_input, "resnet_power_black_r%d.csv" % rep, "Black r%d/%d" % (rep, N_REPS))

# ===== White pixels =====
print("\n" + "=" * 60)
print("WHITE PIXELS (all ones) — %d reps" % N_REPS)
print("=" * 60, flush=True)

white_input = np.ones((1, 3, 224, 224), dtype=np.float32)
for rep in range(N_REPS):
    run_rep(white_input, "resnet_power_white_r%d.csv" % rep, "White r%d/%d" % (rep, N_REPS))

# ===== Final summary =====
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60, flush=True)
csvs = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
for cls in ['black', 'white']:
    files = [f for f in csvs if cls in f]
    if files:
        means = []
        for f in files[:3]:
            d = np.loadtxt(os.path.join(OUTPUT_DIR, f), delimiter=',', skiprows=1)
            ane_idx = list(open(os.path.join(OUTPUT_DIR, f)).readline().strip().split(',')).index('ANE')
            means.append(d[300:, ane_idx].mean())
        print("%s: %d files, sample ANE mean=%.1f" % (cls, len(files), np.mean(means)), flush=True)
print("Done!", flush=True)
