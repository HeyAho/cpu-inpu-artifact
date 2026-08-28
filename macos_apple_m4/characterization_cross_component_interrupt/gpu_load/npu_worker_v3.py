#!/usr/bin/env python3
"""NPU worker v3 - 统一使用 ResNet50 B8 S224, normal 加 throttle"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import coremltools as ct

ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(
    ARCHIVE_ROOT
    / "models/by_platform/macos_apple_m4/characterization_cross_component_interrupt"
    / "ResNet50_B8_Size224.mlpackage"
)
BATCH_SIZE = 8
IMAGE_SIZE = 224

def run_npu_loop(level, duration):
    model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
    input_name = model.input_description._fd_spec[0].name
    data = {input_name: np.random.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)}

    throttle = 0.005 if level == "normal" else 0  # 50ms sleep between iters for normal

    print(f"[NPU] {level}: B={BATCH_SIZE} S={IMAGE_SIZE}, duration={duration}s, throttle={throttle}s")

    _ = model.predict(data)  # warmup

    print(f"[NPU] Running...")
    loop_count = 0
    start_time = time.time()
    while (time.time() - start_time) < duration:
        _ = model.predict(data)
        loop_count += 1
        if throttle > 0:
            time.sleep(throttle)
    actual = time.time() - start_time
    print(f"[NPU] Done: {loop_count} iters in {actual:.1f}s ({loop_count/actual:.1f}/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["normal", "full"])
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()
    run_npu_loop(args.level, args.duration)
