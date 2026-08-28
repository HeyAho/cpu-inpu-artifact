#!/usr/bin/env python3
"""
GPU worker v2 (MPS): GPU workload via PyTorch MPS (Metal Performance Shaders)
  Full:  matmul N=2048 (saturates GPU)
  Normal: matmul N=128 (light GPU load)
Replaces pyopencl with torch.device('mps') for Apple Silicon.
"""
import torch
import time
import argparse

def run_matmul_loop(N, duration):
    device = torch.device("mps")
    mb = (N * N * 4) / (1024 * 1024)
    print(f"[GPU] MPS device, Matrix: {N}x{N}, {mb*3:.1f}MB total, duration={duration}s")

    A = torch.rand(N, N, device=device, dtype=torch.float32)
    B = torch.rand(N, N, device=device, dtype=torch.float32)

    # Pre-warm the MPS pipeline
    _ = torch.mm(A, B)
    torch.mps.synchronize()

    print(f"[GPU] Running...")
    loop_count = 0
    start_time = time.time()

    # Small kernel (N=128) needs batching to saturate GPU
    batch = 1 if N >= 2048 else 100

    while True:
        for _ in range(batch):
            C = torch.mm(A, B)
            loop_count += 1
        torch.mps.synchronize()
        if (time.time() - start_time) >= duration:
            break

    actual = time.time() - start_time
    print(f"[GPU] Done: {loop_count} iters in {actual:.1f}s ({loop_count/actual:.1f}/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["normal", "full"], help="GPU load level")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    args = parser.parse_args()
    N = 2048 if args.level == "full" else 128
    run_matmul_loop(N, args.duration)
