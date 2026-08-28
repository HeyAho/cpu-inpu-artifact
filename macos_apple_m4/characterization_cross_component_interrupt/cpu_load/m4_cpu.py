#!/usr/bin/env python3
"""
CPU worker for NPU-CPU cross-power experiments.
Targets P-cores or E-cores via pthread QoS.
Levels: normal (2 cores), full (4 cores).
"""
import multiprocessing, time, sys, argparse, ctypes, ctypes.util

# QoS constants (Darwin)
QOS_CLASS_USER_INITIATED = 0x21  # P-cores
QOS_CLASS_BACKGROUND     = 0x09  # E-cores

# Cache libSystem handle
_libsystem = None

def set_thread_qos(qos_class):
    """Set QoS class for current thread."""
    global _libsystem
    if _libsystem is None:
        _libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
    ret = _libsystem.pthread_set_qos_class_self_np(qos_class, 0)
    if ret != 0:
        print(f"[WARN] pthread_set_qos_class_self_np({qos_class}) returned {ret}")

def calc_prime_interval(start, count):
    """CPU-bound: check primality of numbers (unpredictable branches)."""
    found = 0
    n = start
    while found < count:
        n += 1
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            found += 1
    return n

def compute_loop(duration, qos_class):
    """Run compute for `duration` seconds on specified QoS core type."""
    set_thread_qos(qos_class)
    end = time.time() + duration
    n = 2
    while time.time() < end:
        # Find next 100 primes, then a burst of FP math
        n = calc_prime_interval(n, 100)
        x = 0.0
        for _ in range(50000):
            x += (3.14159 ** 0.5) / (2.71828 ** 0.5 + x * 1e-10)
        _ = x  # suppress optimizer
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("core", choices=["p", "e"])
    parser.add_argument("level", choices=["normal", "full"])
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()

    n_cores = 2 if args.level == "normal" else 4
    qos = QOS_CLASS_USER_INITIATED if args.core == "p" else QOS_CLASS_BACKGROUND
    label = "P-core" if args.core == "p" else "E-core"

    print(f"[CPU] {label} {args.level}: {n_cores} core(s), QoS={qos:#x}, {args.duration}s")

    procs = []
    for _ in range(n_cores):
        p = multiprocessing.Process(target=compute_loop, args=(args.duration, qos))
        p.start()
        procs.append(p)

    t0 = time.time()
    for p in procs:
        p.join(timeout=args.duration + 60)
        if p.is_alive():
            p.terminate()
            p.join()

    print(f"[CPU] Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
