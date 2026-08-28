#!/usr/bin/env python3
"""
GPU worker v2: 使用用户指定的精确 GPU 工作负载
  Full:  run_matmul_loop(N=2048)
  Normal: run_matmul_loop(N=128)
"""
import pyopencl as cl
import numpy as np
import time
import sys
import argparse

def get_gpu_context():
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            print(f"[GPU] {devices[0].name}")
            return cl.Context(devices=[devices[0]]), devices[0]
    print("[GPU] Not found!")
    sys.exit(1)

def run_matmul_loop(N, duration):
    ctx, device = get_gpu_context()
    queue = cl.CommandQueue(ctx)

    mb = (N * N * 4) / (1024 * 1024)
    print(f"[GPU] Matrix: {N}x{N}, {mb*3:.1f}MB total, duration={duration}s")

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N)).astype(np.float32)

    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    kernel_code = """
    __kernel void matmul(const int N, __global float* A, __global float* B, __global float* C) {
        int i = get_global_id(0), j = get_global_id(1);
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[i * N + k] + B[k * N + j];
        C[i * N + j] = sum;
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    print(f"[GPU] Running...")
    loop_count = 0
    start_time = time.time()
    while (time.time() - start_time) < duration:
        prg.matmul(queue, (N, N), None, np.int32(N), A_buf, B_buf, C_buf)
        queue.finish()
        loop_count += 1
    actual = time.time() - start_time
    print(f"[GPU] Done: {loop_count} iters in {actual:.1f}s ({loop_count/actual:.1f}/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["normal", "full"], help="GPU load level")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    args = parser.parse_args()
    N = 2048 if args.level == "full" else 128
    run_matmul_loop(N, args.duration)
