#!/usr/bin/env python3
import argparse
import time

import numpy as np
import pyopencl as cl


KERNEL_SOURCE = r"""
__kernel void matmul(
    const int n,
    __global const float *a,
    __global const float *b,
    __global float *c)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    float value = 0.0f;
    for (int k = 0; k < n; ++k) {
        value += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = value;
}
"""


def gpu_context():
    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            return cl.Context(devices=[devices[0]]), devices[0]
    raise RuntimeError("No OpenCL GPU device found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty-cycle", type=float, required=True)
    parser.add_argument("--matrix-size", type=int, default=192)
    parser.add_argument("--cycle-ms", type=float, default=200.0)
    args = parser.parse_args()

    duty = min(100.0, max(0.0, args.duty_cycle)) / 100.0
    context, device = gpu_context()
    queue = cl.CommandQueue(context)
    program = cl.Program(context, KERNEL_SOURCE).build()
    kernel = cl.Kernel(program, "matmul")

    size = args.matrix_size
    host_a = np.random.default_rng(20260711).random((size, size), dtype=np.float32)
    host_b = np.random.default_rng(20260712).random((size, size), dtype=np.float32)
    flags = cl.mem_flags
    device_a = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=host_a)
    device_b = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=host_b)
    device_c = cl.Buffer(context, flags.WRITE_ONLY, host_a.nbytes)
    kernel.set_args(np.int32(size), device_a, device_b, device_c)

    print(f"GPU_READY device={device.name} duty={duty * 100:.0f}% size={size}", flush=True)
    cycle_s = args.cycle_ms / 1000.0
    while True:
        cycle_start = time.monotonic()
        active_until = cycle_start + cycle_s * duty
        while time.monotonic() < active_until:
            cl.enqueue_nd_range_kernel(queue, kernel, (size, size), None)
            queue.finish()
        sleep_s = cycle_s - (time.monotonic() - cycle_start)
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
