#!/usr/bin/env python3
import argparse
import mmap
import os
import signal
import struct
from pathlib import Path


VENV = Path(os.environ.get("RYZEN_AI_VENV", Path(__file__).resolve().parent.parent / ".venv"))


def configure_environment():
    paths = [
        VENV / "deployment/lib",
        VENV / "lib/python3.12/site-packages/voe/lib",
        VENV / "lib/python3.12/site-packages/onnxruntime/capi",
        Path("/opt/xilinx/xrt/lib"),
        Path("/lib/x86_64-linux-gnu"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(str(path) for path in paths) + (f":{existing}" if existing else "")
    os.environ["XILINX_XRT"] = "/opt/xilinx/xrt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--counter-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    configure_environment()
    import numpy as np
    import onnxruntime as ort

    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        args.model,
        sess_options=session_options,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    )
    if session.get_providers()[0] != "VitisAIExecutionProvider":
        raise RuntimeError(f"VitisAI EP is not primary: {session.get_providers()}")
    input_info = session.get_inputs()[0]
    shape = tuple(1 if isinstance(value, str) else value for value in input_info.shape)
    input_data = np.random.default_rng(20260712).standard_normal(shape, dtype=np.float32)
    session.run(None, {input_info.name: input_data})
    counter_path = Path(args.counter_file)
    counter_path.write_bytes(b"\0" * 8)
    with counter_path.open("r+b") as handle:
        counter = mmap.mmap(handle.fileno(), 8)
        Path(args.ready_file).write_text("ready\n")
        print(f"NPU_READY providers={session.get_providers()} model={args.model} input={shape}", flush=True)
        count = 0
        stop_file = Path(args.stop_file)
        while running and not stop_file.exists():
            session.run(None, {input_info.name: input_data})
            count += 1
            counter.seek(0)
            counter.write(struct.pack("<Q", count))
        counter.flush()
        counter.close()


if __name__ == "__main__":
    main()
