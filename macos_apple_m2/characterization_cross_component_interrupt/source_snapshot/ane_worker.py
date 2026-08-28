#!/usr/bin/env python3
import argparse
import mmap
import struct
import time
from pathlib import Path

import coremltools as ct
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--counter-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--target-rate", type=float, default=100.0)
    args = parser.parse_args()
    model = ct.models.CompiledMLModel(args.model, compute_units=ct.ComputeUnit.CPU_AND_NE)
    input_data = np.random.default_rng(20260712).random((1, 3, 224, 224), dtype=np.float32)
    model.predict({"input_tensor": input_data})
    counter_path = Path(args.counter_file)
    counter_path.write_bytes(b"\0" * 8)
    with counter_path.open("r+b") as handle:
        counter = mmap.mmap(handle.fileno(), 8)
        Path(args.ready_file).write_text("ready\n")
        print(f"ANE_READY model={args.model}", flush=True)
        count = 0
        minimum_cycle = 1.0 / args.target_rate if args.target_rate > 0 else 0.0
        while not Path(args.stop_file).exists():
            cycle_start = time.monotonic()
            model.predict({"input_tensor": input_data})
            count += 1
            counter.seek(0)
            counter.write(struct.pack("<Q", count))
            remaining = minimum_cycle - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)
        counter.flush()
        counter.close()


if __name__ == "__main__":
    main()
