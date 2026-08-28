#!/usr/bin/env python3
"""Line-oriented OpenVINO NPU inference server for Experiment 1."""

from __future__ import annotations

import argparse
import mmap
import os
import sys

import numpy as np
import openvino as ov


def reply(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("shared_path")
    parser.add_argument("shared_bytes", type=int)
    parser.add_argument("shared_dev", type=int)
    parser.add_argument("shared_ino", type=int)
    args = parser.parse_args()

    core = ov.Core()
    if "NPU" not in core.available_devices:
        raise RuntimeError(f"OpenVINO NPU is unavailable: {core.available_devices}")

    compiled = core.compile_model(args.model, "NPU")
    execution_devices = str(compiled.get_property("EXECUTION_DEVICES"))
    if "NPU" not in execution_devices.upper():
        raise RuntimeError(f"model was not placed on NPU: {execution_devices}")

    request = compiled.create_infer_request()
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
    ov_input = ov.Tensor(input_array, shared_memory=True)
    ov_pointer = np.asarray(ov_input.data).__array_interface__["data"][0]
    if ov_pointer != host_pointer:
        raise RuntimeError(
            f"OpenVINO copied the host input: {host_pointer:#x} != {ov_pointer:#x}"
        )
    request.set_input_tensor(ov_input)

    # Prove that modifying the mapped pages after tensor binding changes the
    # model result. Restore the C program's original bytes before READY.
    original_input = input_array.copy()
    input_array.fill(0.0)
    request.infer()
    zero_output = np.array(request.get_output_tensor(0).data, copy=True)
    input_array.fill(1.0)
    request.infer()
    one_output = np.array(request.get_output_tensor(0).data, copy=True)
    np.copyto(input_array, original_input)
    if not np.any(zero_output != one_output):
        raise RuntimeError("bound shared-memory updates did not reach the model")

    # Compile/load and coherence-test overhead is outside the measured experiment.
    request.infer()
    reply(
        "READY backend=openvino "
        f"device={execution_devices} shared_memory=verified "
        f"dev={status.st_dev} ino={status.st_ino} host_ptr={host_pointer:#x} "
        f"coherence_delta={np.max(np.abs(one_output - zero_output)):.9g}"
    )

    for raw_line in sys.stdin:
        fields = raw_line.strip().split()
        if not fields:
            continue
        if fields[0] == "quit":
            reply("BYE")
            # Some Intel NPU plugin builds can block in Python interpreter
            # teardown after all requests have completed. The parent has
            # already received BYE, so bypass plugin destructors.
            os._exit(0)
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
            request.infer()
            output = request.get_output_tensor(0).data
            checksum += float(np.asarray(output).reshape(-1)[0])
        reply(f"OK count={count} checksum={checksum:.9g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
