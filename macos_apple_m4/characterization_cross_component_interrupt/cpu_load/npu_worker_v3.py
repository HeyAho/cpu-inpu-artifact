#!/usr/bin/env python3
"""NPU worker v3 - 直接使用原始脚本的模型和配置，仅添加 duration 控制"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import coremltools as ct

ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
M4_CROSS_COMPONENT_MODEL = str(
    ARCHIVE_ROOT
    / "models/by_platform/macos_apple_m4/characterization_cross_component_interrupt"
    / "ResNet50_B8_Size224.mlpackage"
)

# NPU Full: WideResNet101_2 B=16 Size=512 (same as WideResNet101_2.py)
NPU_FULL_MODEL = M4_CROSS_COMPONENT_MODEL
NPU_FULL_BATCH = 8
NPU_FULL_SIZE = 224

# NPU Normal: ResNet50 B=4 Size=16 (same as resnet50.-npu-batch32size.py)
NPU_NORMAL_MODEL = M4_CROSS_COMPONENT_MODEL
NPU_NORMAL_BATCH = 4
NPU_NORMAL_SIZE = 16

def run_npu_loop(model_path, batch_size, image_size, duration):
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    input_name = model.input_description._fd_spec[0].name
    data = {input_name: np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)}

    print(f"[NPU] Model: {os.path.basename(model_path)}, B={batch_size} S={image_size}, duration={duration}s")

    # Warmup
    _ = model.predict(data)

    print(f"[NPU] Running...")
    loop_count = 0
    start_time = time.time()
    while (time.time() - start_time) < duration:
        _ = model.predict(data)
        loop_count += 1
    actual = time.time() - start_time
    print(f"[NPU] Done: {loop_count} iters in {actual:.1f}s ({loop_count/actual:.1f}/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["normal", "full"])
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()

    if args.level == "full":
        run_npu_loop(NPU_FULL_MODEL, NPU_FULL_BATCH, NPU_FULL_SIZE, args.duration)
    else:
        run_npu_loop(NPU_NORMAL_MODEL, NPU_NORMAL_BATCH, NPU_NORMAL_SIZE, args.duration)
