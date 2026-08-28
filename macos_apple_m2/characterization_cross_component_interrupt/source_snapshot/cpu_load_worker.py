#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import multiprocessing as mp
import signal
import time


RUNNING = True
QOS_CLASS_USER_INITIATED = 0x21


def stop(signum, frame):
    global RUNNING
    RUNNING = False


def worker(duty_fraction):
    global RUNNING
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
    libsystem.pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)
    cycle_s = 0.1
    value = 1
    while RUNNING:
        cycle_start = time.monotonic()
        active_until = cycle_start + cycle_s * duty_fraction
        while time.monotonic() < active_until:
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
            value ^= value >> 13
        remaining = cycle_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty-cycle", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    duty = min(100.0, max(0.0, args.duty_cycle)) / 100.0
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(duty,)) for _ in range(args.workers)]
    for process in processes:
        process.start()
    print(f"CPU_READY duty={duty * 100:.0f}% workers={len(processes)}", flush=True)
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
