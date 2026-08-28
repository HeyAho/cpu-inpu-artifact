import subprocess, time, os, signal
from pathlib import Path
import numpy as np
import coremltools as ct

ROOT = Path(__file__).resolve().parent
MONITOR = os.environ.get("INPU_MONITOR", "/tmp/neo_monitor_m2")
MODEL = os.environ.get(
    "INPU_RESNET50_MODEL",
    str(ROOT / "../../models/by_platform/macos_apple_m2/characterization_power/ResNet50_ANE.mlpackage"),
)
OUT = os.environ.get("INPU_OUTPUT_DIR", str(ROOT / "data"))
N_REPS = 30
WARMUP = 50

def kill():
    for n in ["neo_monitor","python3"]:
        try: subprocess.run(["pkill","-9","-f",n],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=3)
        except: pass
    time.sleep(1)

model = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.CPU_AND_NE)
Path(OUT).mkdir(parents=True, exist_ok=True)
iname = model.input_description._fd_spec[0].name
print("Input: %s" % iname)

# Warmup
data = {iname: np.zeros((1,3,224,224), dtype=np.float32)}
for _ in range(WARMUP): model.predict(data)
print("Warmup done")

for label, arr in [("black", np.zeros((1,3,224,224), dtype=np.float32)),
                    ("white", np.ones((1,3,224,224), dtype=np.float32))]:
    for rep in range(N_REPS):
        kill()
        csv_name = "resnet_power_%s_r%d.csv" % (label, rep)
        csv_path = os.path.join(OUT, csv_name)
        if os.path.exists(csv_path):
            # Skip if already exists and has data
            if os.path.getsize(csv_path) > 5000: continue
            os.remove(csv_path)
        
        mon = subprocess.Popen([MONITOR, csv_name], cwd=OUT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        time.sleep(5)  # baseline
        
        start = time.time(); iters = 0
        while time.time() - start < 15:
            model.predict({iname: arr}); iters += 1
        elapsed = time.time() - start
        
        time.sleep(3)
        mon.send_signal(signal.SIGTERM)
        try: mon.wait(timeout=5)
        except: mon.kill()
        time.sleep(1)
        
        size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
        print("%s r%d: %d iters %.1fs (%.0f/s) CSV=%dKB" % (label, rep, iters, elapsed, iters/elapsed, size//1024))
        kill()

print("Done!")
