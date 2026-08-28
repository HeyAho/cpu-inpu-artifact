#!/usr/bin/env python3
import argparse
import mmap
import signal
import struct
from pathlib import Path

import numpy as np
import openvino as ov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--counter-file", required=True)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    core = ov.Core()
    model = core.read_model(args.model)
    model.reshape([1, 224, 224, 3])
    compiled = core.compile_model(model, "NPU")
    output = compiled.output(0)
    input_data = np.random.default_rng(20260712).standard_normal(
        (1, 224, 224, 3), dtype=np.float32
    )
    counter_path = Path(args.counter_file)
    counter_path.write_bytes(b"\0" * 8)
    with counter_path.open("r+b") as handle:
        counter = mmap.mmap(handle.fileno(), 8)
        Path(args.ready_file).write_text("ready\n")
        count = 0
        while running:
            compiled([input_data])[output]
            count += 1
            counter.seek(0)
            counter.write(struct.pack("<Q", count))
        counter.flush()
        counter.close()


if __name__ == "__main__":
    main()
