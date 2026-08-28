#!/usr/bin/env python3
"""Line-oriented ONNX Runtime/Vitis AI NPU server for Experiment 1."""

from __future__ import annotations

import argparse
import mmap
import os
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ACTIVE_TIME = Path("/sys/class/accel/accel0/device/power/runtime_active_time")


def reply(message: str) -> None:
    print(message, flush=True)


def npu_active_time() -> int:
    return int(ACTIVE_TIME.read_text(encoding="ascii").strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("shared_path")
    parser.add_argument("shared_bytes", type=int)
    parser.add_argument("shared_dev", type=int)
    parser.add_argument("shared_ino", type=int)
    args = parser.parse_args()

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"Vitis AI EP is unavailable: {ort.get_available_providers()}"
        )
    if not Path("/dev/accel/accel0").exists():
        raise RuntimeError("/dev/accel/accel0 is unavailable")

    session = ort.InferenceSession(
        args.model, providers=["VitisAIExecutionProvider"]
    )
    if session.get_providers()[0] != "VitisAIExecutionProvider":
        raise RuntimeError(f"unexpected provider order: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    shared_file = open(args.shared_path, "r+b", buffering=0)
    status = os.fstat(shared_file.fileno())
    if (status.st_dev, status.st_ino) != (args.shared_dev, args.shared_ino):
        raise RuntimeError(
            "shared-memory identity mismatch: "
            f"got {(status.st_dev, status.st_ino)}, "
            f"expected {(args.shared_dev, args.shared_ino)}"
        )
    shared_map = mmap.mmap(
        shared_file.fileno(), args.shared_bytes, access=mmap.ACCESS_WRITE
    )
    input_array = np.ndarray(
        (1, 3, 256, 256), dtype=np.float32, buffer=shared_map
    )
    host_pointer = input_array.__array_interface__["data"][0]
    ort_input = ort.OrtValue.ortvalue_from_numpy(input_array)
    if ort_input.data_ptr() != host_pointer:
        raise RuntimeError(
            "ONNX Runtime copied the host input: "
            f"{host_pointer:#x} != {ort_input.data_ptr():#x}"
        )
    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(input_name, ort_input)
    io_binding.bind_output(output_name, "cpu")

    def run_once() -> np.ndarray:
        session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]

    # Prove that modifications made after binding reach the model, then restore
    # the C program's original bytes before READY.
    original_input = input_array.copy()
    active_before = npu_active_time()
    input_array.fill(0.0)
    zero_output = np.array(run_once(), copy=True)
    input_array.fill(1.0)
    one_output = np.array(run_once(), copy=True)
    np.copyto(input_array, original_input)
    output = run_once()
    active_after = npu_active_time()
    if not np.any(zero_output != one_output):
        raise RuntimeError("bound shared-memory updates did not reach the model")

    # The EP can register successfully while executing an unsupported graph on
    # CPU. Require the kernel driver's active-time counter to advance.
    if active_after <= active_before:
        raise RuntimeError(
            "Vitis AI session did not advance the AMD NPU active-time counter"
        )

    reply(
        "READY backend=onnxruntime device=AMD-XDNA "
        "shared_memory=verified "
        f"dev={status.st_dev} ino={status.st_ino} host_ptr={host_pointer:#x} "
        f"active_delta_us={active_after - active_before} "
        f"coherence_delta={np.max(np.abs(one_output - zero_output)):.9g} "
        f"warmup_checksum={float(np.asarray(output).reshape(-1)[0]):.9g}"
    )

    for raw_line in sys.stdin:
        fields = raw_line.strip().split()
        if not fields:
            continue
        if fields[0] == "quit":
            reply("BYE")
            return 0
        if fields[0] != "infer" or len(fields) != 2:
            reply("ERROR expected: infer N")
            continue
        try:
            count = int(fields[1])
            if count < 1:
                raise ValueError
        except ValueError:
            reply("ERROR invalid inference count")
            continue

        checksum = 0.0
        for _ in range(count):
            output = run_once()
            checksum += float(np.asarray(output).reshape(-1)[0])
        reply(f"OK count={count} checksum={checksum:.9g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
