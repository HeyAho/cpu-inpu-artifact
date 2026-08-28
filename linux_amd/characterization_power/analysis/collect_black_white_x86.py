#!/usr/bin/env python3
import argparse
import csv
import ctypes
import glob
import hashlib
import json
import os
import re
import signal
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


INTEL_PMT_GUID = "0x130670b2"
INTEL_ENERGY_OFFSET = 0x628
INTEL_WORKPOINT_OFFSET = 0x68
INTEL_ENERGY_SCALE = 16384.0
INTEL_NPU_BDF = "0000:00:0b.0"
AMD_NPU_BDF = "0000:05:00.1"


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def read_int(path, default=0):
    value = read_text(path)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def count_irqs(pci_id, keyword):
    cpu_count = os.cpu_count() or 1
    total = 0
    vectors = 0
    try:
        lines = Path("/proc/interrupts").read_text().splitlines()
    except OSError:
        return 0, 0
    for line in lines:
        if pci_id not in line or keyword not in line:
            continue
        fields = line.split()
        values = [int(value) for value in fields[1:1 + cpu_count] if value.isdigit()]
        if values:
            total += sum(values)
            vectors += 1
    return total, vectors


class MetricsHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("structure_size", ctypes.c_uint16),
        ("format_revision", ctypes.c_uint8),
        ("content_revision", ctypes.c_uint8),
    ]


class GPUMetricsV30(ctypes.LittleEndianStructure):
    _fields_ = [
        ("common_header", MetricsHeader),
        ("temperature_gfx", ctypes.c_uint16),
        ("temperature_soc", ctypes.c_uint16),
        ("temperature_core", ctypes.c_uint16 * 16),
        ("temperature_skin", ctypes.c_uint16),
        ("average_gfx_activity", ctypes.c_uint16),
        ("average_vcn_activity", ctypes.c_uint16),
        ("average_ipu_activity", ctypes.c_uint16 * 8),
        ("average_core_c0_activity", ctypes.c_uint16 * 16),
        ("average_dram_reads", ctypes.c_uint16),
        ("average_dram_writes", ctypes.c_uint16),
        ("average_ipu_reads", ctypes.c_uint16),
        ("average_ipu_writes", ctypes.c_uint16),
        ("system_clock_counter", ctypes.c_uint64),
        ("average_socket_power", ctypes.c_uint32),
        ("average_ipu_power", ctypes.c_uint16),
        ("average_apu_power", ctypes.c_uint32),
        ("average_gfx_power", ctypes.c_uint32),
        ("average_dgpu_power", ctypes.c_uint32),
        ("average_all_core_power", ctypes.c_uint32),
        ("average_core_power", ctypes.c_uint16 * 16),
        ("average_sys_power", ctypes.c_uint16),
        ("stapm_power_limit", ctypes.c_uint16),
        ("current_stapm_power_limit", ctypes.c_uint16),
        ("average_gfxclk_frequency", ctypes.c_uint16),
        ("average_socclk_frequency", ctypes.c_uint16),
        ("average_vpeclk_frequency", ctypes.c_uint16),
        ("average_ipuclk_frequency", ctypes.c_uint16),
        ("average_fclk_frequency", ctypes.c_uint16),
        ("average_vclk_frequency", ctypes.c_uint16),
        ("average_uclk_frequency", ctypes.c_uint16),
        ("average_mpipu_frequency", ctypes.c_uint16),
        ("current_coreclk", ctypes.c_uint16 * 16),
        ("current_core_maxfreq", ctypes.c_uint16),
        ("current_gfx_maxfreq", ctypes.c_uint16),
        ("throttle_residency_prochot", ctypes.c_uint32),
        ("throttle_residency_spl", ctypes.c_uint32),
        ("throttle_residency_fppt", ctypes.c_uint32),
        ("throttle_residency_sppt", ctypes.c_uint32),
        ("throttle_residency_thm_core", ctypes.c_uint32),
        ("throttle_residency_thm_gfx", ctypes.c_uint32),
        ("throttle_residency_thm_soc", ctypes.c_uint32),
        ("time_filter_alphavalue", ctypes.c_uint32),
    ]


def find_amd_gpu_metrics():
    expected = ctypes.sizeof(GPUMetricsV30)
    for path in sorted(glob.glob("/sys/class/drm/card*/device/gpu_metrics")):
        try:
            raw = Path(path).read_bytes()[:expected]
            metrics = GPUMetricsV30.from_buffer_copy(raw)
        except Exception:
            continue
        header = metrics.common_header
        if header.structure_size == expected and header.format_revision == 3:
            return Path(path)
    raise RuntimeError("AMD gpu_metrics v3.0 endpoint not found")


def read_amd_gpu_metrics(path):
    expected = ctypes.sizeof(GPUMetricsV30)
    raw = path.read_bytes()[:expected]
    metrics = GPUMetricsV30.from_buffer_copy(raw)
    ipu_activity = list(metrics.average_ipu_activity)
    return {
        "npu_power_w": metrics.average_ipu_power / 1000.0,
        "npu_freq_mhz": float(metrics.average_ipuclk_frequency),
        "npu_secondary_freq_mhz": float(metrics.average_mpipu_frequency),
        "npu_activity_pct": sum(ipu_activity) / len(ipu_activity),
        "npu_reads_mb_s": float(metrics.average_ipu_reads),
        "npu_writes_mb_s": float(metrics.average_ipu_writes),
        "socket_power_w": metrics.average_socket_power / 1000.0,
        "apu_power_w": metrics.average_apu_power / 1000.0,
        "gfx_power_w": metrics.average_gfx_power / 1000.0,
        "metrics_filter_us": metrics.time_filter_alphavalue,
    }


class IntelSampler:
    def __init__(self):
        self.telem = self._find_telem()
        self.npu_sysfs = self._find_npu_sysfs()
        self.previous = None

    def _find_telem(self):
        for path in sorted(glob.glob("/sys/class/intel_pmt/telem*")):
            if read_text(Path(path) / "guid") == INTEL_PMT_GUID:
                return Path(path) / "telem"
        raise RuntimeError("Intel PMT NPU telemetry not found")

    def _find_npu_sysfs(self):
        for path in sorted(glob.glob("/sys/bus/pci/drivers/intel_vpu/0000:*")):
            if Path(path).is_dir():
                return Path(path)
        raise RuntimeError("Intel NPU sysfs directory not found")

    def snapshot(self):
        now = time.monotonic()
        buffer = self.telem.read_bytes()
        energy_j = struct.unpack_from("<Q", buffer, INTEL_ENERGY_OFFSET)[0] / INTEL_ENERGY_SCALE
        workpoint = struct.unpack_from("<I", buffer, INTEL_WORKPOINT_OFFSET)[0]
        irq_total, irq_vectors = count_irqs(INTEL_NPU_BDF, "intel_vpu")
        return {
            "time": now,
            "energy_j": energy_j,
            "freq_mhz": (workpoint & 0xFF) * (100.0 / 3.0),
            "busy_us": read_int(self.npu_sysfs / "npu_busy_time_us"),
            "irq_total": irq_total,
            "irq_vectors": irq_vectors,
        }

    def reset(self):
        self.previous = self.snapshot()

    def sample(self, work_count):
        current = self.snapshot()
        previous = self.previous or current
        self.previous = current
        elapsed = max(1e-9, current["time"] - previous["time"])
        return {
            "npu_power_w": max(0.0, current["energy_j"] - previous["energy_j"]) / elapsed,
            "npu_freq_mhz": current["freq_mhz"],
            "npu_activity_pct": min(100.0, max(0.0, (current["busy_us"] - previous["busy_us"]) / (elapsed * 1e6) * 100.0)),
            "npu_irq_per_s": max(0, current["irq_total"] - previous["irq_total"]) / elapsed,
            "npu_irq_vectors": current["irq_vectors"],
            "npu_work_per_s": work_count / elapsed,
        }


class AmdSampler:
    def __init__(self):
        self.metrics_path = find_amd_gpu_metrics()
        self.previous = None

    def snapshot(self):
        now = time.monotonic()
        irq_total, irq_vectors = count_irqs(AMD_NPU_BDF, "xdna")
        metrics = read_amd_gpu_metrics(self.metrics_path)
        metrics.update({"time": now, "irq_total": irq_total, "irq_vectors": irq_vectors})
        return metrics

    def reset(self):
        self.previous = self.snapshot()

    def sample(self, work_count):
        current = self.snapshot()
        previous = self.previous or current
        self.previous = current
        elapsed = max(1e-9, current["time"] - previous["time"])
        row = dict(current)
        row.pop("time", None)
        row.pop("irq_total", None)
        row.pop("irq_vectors", None)
        row["npu_irq_per_s"] = max(0, current["irq_total"] - previous["irq_total"]) / elapsed
        row["npu_irq_vectors"] = current["irq_vectors"]
        row["npu_work_per_s"] = work_count / elapsed
        return row


class IntelRunner:
    def __init__(self, model_path):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled = core.compile_model(model, "NPU")
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        self.shape = tuple(self.input_port.shape)
        if self.shape != (1, 3, 224, 224):
            raise RuntimeError(f"Expected aligned NCHW input (1, 3, 224, 224), got {self.shape}")
        self.backend = "OpenVINO NPU"
        self.execution_providers = ["NPU"]

    def make_input(self, value):
        return np.full(self.shape, value, dtype=np.float32)

    def infer(self, array):
        self.compiled({self.input_port: array})[self.output_port]


class AmdRunner:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["VitisAIExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.shape = tuple(int(dim) for dim in self.input.shape)
        if self.shape != (1, 3, 224, 224):
            raise RuntimeError(f"Expected aligned NCHW input (1, 3, 224, 224), got {self.shape}")
        self.backend = "ONNX Runtime VitisAIExecutionProvider"
        self.execution_providers = self.session.get_providers()

    def make_input(self, value):
        return np.full(self.shape, value, dtype=np.float32)

    def infer(self, array):
        self.session.run(None, {self.input.name: array})


def run_inference_loop(runner, array, stop_event, counter):
    while not stop_event.is_set():
        runner.infer(array)
        counter[0] += 1


def collect_phase(writer, sampler, phase, seconds, interval, counter, platform, label, rep):
    deadline = time.monotonic() + seconds
    prev_count = counter[0]
    while time.monotonic() < deadline:
        time.sleep(interval)
        current_count = counter[0]
        row = sampler.sample(current_count - prev_count)
        prev_count = current_count
        row.update({
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "platform": platform,
            "label": label,
            "rep": rep,
            "phase": phase,
        })
        writer.writerow(row)


def run_trial(runner, sampler, out_csv, platform, label, rep, array, args):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counter = [0]
    sampler.reset()
    fieldnames = None
    with out_csv.open("w", newline="") as handle:
        writer = None

        def write_row(row):
            nonlocal fieldnames, writer
            if fieldnames is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

        class DynamicWriter:
            def writerow(self, row):
                write_row(row)

        dyn_writer = DynamicWriter()
        collect_phase(dyn_writer, sampler, "baseline", args.baseline, args.interval, counter, platform, label, rep)
        stop_event = threading.Event()
        worker = threading.Thread(target=run_inference_loop, args=(runner, array, stop_event, counter), daemon=True)
        worker.start()
        collect_phase(dyn_writer, sampler, "infer", args.duration, args.interval, counter, platform, label, rep)
        stop_event.set()
        worker.join(timeout=10)
        collect_phase(dyn_writer, sampler, "cooldown", args.cooldown, args.interval, counter, platform, label, rep)
    return counter[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["intel", "amd"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--baseline", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--model-source", default="pytorch/vision:v0.19.0 resnet50 weights=None")
    parser.add_argument("--precision", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    data_dir = outdir / "data"
    if args.platform == "intel":
        runner = IntelRunner(args.model)
        sampler = IntelSampler()
    else:
        runner = AmdRunner(args.model)
        sampler = AmdSampler()

    black = runner.make_input(0.0)
    white = runner.make_input(1.0)
    for _ in range(args.warmup):
        runner.infer(black)

    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    metadata = {
        "platform": args.platform,
        "model": args.model,
        "model_sha256": model_hash,
        "model_source": args.model_source,
        "precision": args.precision,
        "backend": runner.backend,
        "execution_providers": runner.execution_providers,
        "input_shape": list(black.shape),
        "black_value": 0.0,
        "white_value": 1.0,
        "reps": args.reps,
        "warmup": args.warmup,
        "baseline_s": args.baseline,
        "duration_s": args.duration,
        "cooldown_s": args.cooldown,
        "interval_s": args.interval,
        "started_at": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for label, array in [("black", black), ("white", white)]:
        for rep in range(args.reps):
            csv_path = data_dir / f"resnet_power_{label}_r{rep}.csv"
            if csv_path.exists() and csv_path.stat().st_size > 5000:
                print(f"[skip] {csv_path}", flush=True)
                continue
            print(f"[{args.platform}] {label} r{rep}/{args.reps}", flush=True)
            count = run_trial(runner, sampler, csv_path, args.platform, label, rep, array, args)
            print(f"  -> {count} inferences, csv={csv_path.stat().st_size // 1024} KB", flush=True)

    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "completed"
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
