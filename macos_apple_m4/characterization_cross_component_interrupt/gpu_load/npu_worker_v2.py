#!/usr/bin/env python3
"""
NPU worker v2: 使用用户指定的精确 NPU 工作负载
  Full:  WideResNet101_2, B=16, Size=512 (from WideResNet101_2.py)
  Normal: ResNet50, B=4, Size=16 (from resnet50.-npu-batch32size.py)
"""
import os, sys, time, argparse
import numpy as np
import coremltools as ct
import torch
import torchvision

# Model cache paths (relative to CWD)
MODEL_CACHE = {}

def get_model(level):
    if level == "full":
        model_name = "WideResNet101_2"
        batch_size = 16
        image_size = 512
        model_filename = f"{model_name}_B{batch_size}_S{image_size}.mlpackage"
        if os.path.exists(model_filename):
            print(f"[NPU] Using cached: {model_filename}")
            model = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            print(f"[NPU] Building {model_name} B={batch_size} S={image_size}...")
            weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT
            torch_model = torchvision.models.wide_resnet101_2(weights=weights)
            torch_model.eval()
            example_input = torch.rand(batch_size, 3, image_size, image_size)
            traced_model = torch.jit.trace(torch_model, example_input)
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape, name="input_1")],
                source="pytorch",
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
            model.save(model_filename)
            print(f"[NPU] Saved: {model_filename}")
    else:  # normal
        model_name = "ResNet50"
        batch_size = 4
        image_size = 16
        model_filename = f"{model_name}_B{batch_size}_Size{image_size}.mlpackage"
        if os.path.exists(model_filename):
            print(f"[NPU] Using cached: {model_filename}")
            model = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            print(f"[NPU] Building {model_name} B={batch_size} S={image_size}...")
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            torch_model = torchvision.models.resnet50(weights=weights)
            torch_model.eval()
            example_input = torch.rand(batch_size, 3, image_size, image_size)
            traced_model = torch.jit.trace(torch_model, example_input)
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape, name="input_1")],
                source="pytorch",
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
            model.save(model_filename)
            print(f"[NPU] Saved: {model_filename}")


    return model, batch_size, image_size

def run_npu_loop(level, duration):
    model, batch_size, image_size = get_model(level)
    input_name = model.input_description._fd_spec[0].name
    data = {input_name: np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)}

    print(f"[NPU] {level}: B={batch_size} S={image_size}, duration={duration}s")

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
    parser.add_argument("level", choices=["normal", "full"], help="NPU load level")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    args = parser.parse_args()
    run_npu_loop(args.level, args.duration)
