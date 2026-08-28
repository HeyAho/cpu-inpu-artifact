#!/usr/bin/env python3
import argparse
import signal
import time

import numpy as np
import pyopencl as cl


KERNEL = r"""
__kernel void matmul(const int n, __global const float *a,
                     __global const float *b, __global float *c) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float value = 0.0f;
    for (int k = 0; k < n; ++k)
        value += a[row * n + k] * b[k * n + col];
    c[row * n + col] = value;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty-cycle", type=float, required=True)
    parser.add_argument("--matrix-size", type=int, default=384)
    parser.add_argument("--cycle-ms", type=float, default=200.0)
    args = parser.parse_args()
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    devices = []
    for platform in cl.get_platforms():
        devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
    if not devices:
        raise RuntimeError("No OpenCL GPU found")
    context = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(context)
    kernel = cl.Kernel(cl.Program(context, KERNEL).build(), "matmul")
    size = args.matrix_size
    rng = np.random.default_rng(20260712)
    host_a = rng.random((size, size), dtype=np.float32)
    host_b = rng.random((size, size), dtype=np.float32)
    flags = cl.mem_flags
    device_a = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=host_a)
    device_b = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=host_b)
    device_c = cl.Buffer(context, flags.WRITE_ONLY, host_a.nbytes)
    kernel.set_args(np.int32(size), device_a, device_b, device_c)
    duty = min(100.0, max(0.0, args.duty_cycle)) / 100.0
    cycle_s = args.cycle_ms / 1000.0
    print(f"GPU_READY device={devices[0].name} duty={duty * 100:.0f}% size={size}", flush=True)
    while running:
        start = time.monotonic()
        active_until = start + duty * cycle_s
        while running and time.monotonic() < active_until:
            cl.enqueue_nd_range_kernel(queue, kernel, (size, size), None)
            queue.finish()
        remaining = cycle_s - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
