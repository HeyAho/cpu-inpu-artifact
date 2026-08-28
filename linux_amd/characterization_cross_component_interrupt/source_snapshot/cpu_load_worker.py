#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import signal
import time


RUNNING = True


def stop_worker(signum, frame):
    global RUNNING
    RUNNING = False


def worker(cpu, duty_fraction):
    global RUNNING
    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    try:
        os.sched_setaffinity(0, {cpu})
    except OSError:
        pass
    cycle_s = 0.1
    value = cpu + 1
    while RUNNING:
        cycle_start = time.monotonic()
        active_until = cycle_start + cycle_s * duty_fraction
        while time.monotonic() < active_until:
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
            value ^= value >> 13
        sleep_s = cycle_s - (time.monotonic() - cycle_start)
        if sleep_s > 0:
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty-cycle", type=float, required=True)
    parser.add_argument("--reserve-cpus", type=int, default=2)
    args = parser.parse_args()
    cpu_count = os.cpu_count() or 2
    cpus = list(range(min(args.reserve_cpus, cpu_count - 1), cpu_count))
    duty = min(100.0, max(0.0, args.duty_cycle)) / 100.0
    context = mp.get_context("fork")
    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    processes = [context.Process(target=worker, args=(cpu, duty)) for cpu in cpus]
    for process in processes:
        process.start()
    print(f"CPU_READY duty={duty * 100:.0f}% workers={len(processes)}", flush=True)
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
