#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from gpu_metrics_v3 import read_gpu_metrics


PACKAGE_ENERGY = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
PACKAGE_MAX_ENERGY = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
NPU_ACTIVE_TIME = Path("/sys/class/accel/accel0/device/power/runtime_active_time")
GPU_BUSY = Path("/sys/class/drm/card1/device/gpu_busy_percent")
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
ARCHIVE_ROOT = SCRIPT_DIR.parents[2]
GPU_BINARY = Path(os.environ.get("INPU_GPU_LOAD_BINARY", EXPERIMENT_DIR / "tools" / "vk_selfdriven"))
GPU_SLEEP_MS = {25: 230, 50: 85, 75: 23, 100: 0}
NPU_PCI_ID = "0000:05:00.1"
VENV = Path(os.environ.get("RYZEN_AI_VENV", EXPERIMENT_DIR / ".venv"))
NPU_PYTHON = str(VENV / "bin" / "python")
DEFAULT_MODEL = str(
    ARCHIVE_ROOT
    / "models/by_platform/linux_amd/characterization_cross_component_interrupt"
    / "resnet152_int8.onnx"
)
NPU_LD_PATHS = [
    str(VENV / "deployment/lib"),
    str(VENV / "lib/python3.12/site-packages/voe/lib"),
    str(VENV / "lib/python3.12/site-packages/onnxruntime/capi"),
    "/opt/xilinx/xrt/lib",
    "/lib/x86_64-linux-gnu",
]


def read_int(path, default=0):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return default


def read_cpu_ticks():
    values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def read_npu_irq_counts():
    counts = {}
    cpu_count = os.cpu_count() or 1
    for line in Path("/proc/interrupts").read_text().splitlines():
        if NPU_PCI_ID not in line or "xdna_mailbox" not in line:
            continue
        fields = line.split()
        irq = fields[0].rstrip(":")
        cpu_values = []
        for value in fields[1:1 + cpu_count]:
            if not value.isdigit():
                break
            cpu_values.append(int(value))
        if cpu_values:
            counts[irq] = sum(cpu_values)
    return counts


def read_counter(path):
    try:
        raw = Path(path).read_bytes()[:8]
        return struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
    except OSError:
        return 0


class ManagedProcess:
    def __init__(self, process=None, log_handle=None):
        self.process = process
        self.log_handle = log_handle

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        if self.log_handle:
            self.log_handle.close()


def launch(command, log_path, environment=None):
    log_handle = open(log_path, "a")
    process = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
        env=environment,
    )
    return ManagedProcess(process, log_handle)


def start_npu(script_dir, model, counter_file, ready_file, stop_file, log_path):
    environment = os.environ.copy()
    environment["XILINX_XRT"] = "/opt/xilinx/xrt"
    environment["LD_LIBRARY_PATH"] = ":".join(NPU_LD_PATHS + [environment.get("LD_LIBRARY_PATH", "")])
    return launch(
        [
            "taskset", "-c", "0,1", NPU_PYTHON,
            str(script_dir / "npu_resnet152_worker.py"),
            "--model", model,
            "--counter-file", str(counter_file),
            "--ready-file", str(ready_file),
            "--stop-file", str(stop_file),
        ],
        log_path, environment,
    )


def start_external(domain, level, script_dir, log_path):
    if level == 0:
        return ManagedProcess()
    if domain == "cpu":
        command = [sys.executable, str(script_dir / "cpu_load_worker.py"), "--duty-cycle", str(level)]
    else:
        command = [str(GPU_BINARY), "3600", "50", str(GPU_SLEEP_MS[level])]
    return launch(command, log_path)


class Sampler:
    def __init__(self, counter_file):
        self.counter_file = counter_file
        self.package_max = read_int(PACKAGE_MAX_ENERGY)
        self.last_irq_counts = read_npu_irq_counts()
        self.previous = self._snapshot()

    def _snapshot(self):
        total, idle = read_cpu_ticks()
        return {
            "time": time.monotonic(),
            "package_uj": read_int(PACKAGE_ENERGY),
            "cpu_total": total,
            "cpu_idle": idle,
            "active_ms": read_int(NPU_ACTIVE_TIME),
            "work_count": read_counter(self.counter_file),
        }

    def _delta_energy(self, current, previous):
        delta = current - previous
        if delta < 0 and self.package_max > 0:
            delta += self.package_max
        return max(0, delta)

    def _irq_rate(self, elapsed):
        current = read_npu_irq_counts()
        delta = 0
        for irq, count in current.items():
            if irq in self.last_irq_counts and count >= self.last_irq_counts[irq]:
                delta += count - self.last_irq_counts[irq]
            self.last_irq_counts[irq] = count
        return delta / elapsed, len(current)

    def sample(self):
        metrics = read_gpu_metrics()
        current = self._snapshot()
        previous = self.previous
        self.previous = current
        elapsed = current["time"] - previous["time"]
        cpu_delta = current["cpu_total"] - previous["cpu_total"]
        idle_delta = current["cpu_idle"] - previous["cpu_idle"]
        irq_rate, irq_vectors = self._irq_rate(elapsed)
        metrics.update({
            "platform": "amd_xdna2",
            "workload": "vitisai_resnet152_int8",
            "npu_power_w": metrics["npu_ipu_power_w"],
            "npu_primary_freq_mhz": metrics["npu_ipuclk_mhz"],
            "npu_secondary_freq_mhz": metrics["npu_mpipu_mhz"],
            "npu_activity_pct": metrics["npu_ipu_activity_mean_pct"],
            "gpu_freq_mhz": metrics["gpu_clock_mhz"],
            "package_rapl_power_w": self._delta_energy(current["package_uj"], previous["package_uj"]) / 1e6 / elapsed,
            "package_power_w": self._delta_energy(current["package_uj"], previous["package_uj"]) / 1e6 / elapsed,
            "cpu_total_util_pct": 100.0 * (1.0 - idle_delta / cpu_delta) if cpu_delta > 0 else 0.0,
            "gpu_busy_sysfs_pct": float(read_int(GPU_BUSY)),
            "npu_runtime_active_pct": min(100.0, max(0.0, (current["active_ms"] - previous["active_ms"]) / (elapsed * 1000.0) * 100.0)),
            "npu_work_per_s": max(0, current["work_count"] - previous["work_count"]) / elapsed,
            "npu_irq_per_s": irq_rate,
            "npu_irq_vectors": irq_vectors,
        })
        return metrics


def wait_healthy(process, duration):
    if process and process.poll() is not None:
        raise RuntimeError(f"Workload exited early with code {process.returncode}")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            raise RuntimeError(f"Workload exited early with code {process.returncode}")
        time.sleep(0.2)


def wait_ready(process, ready_file, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"NPU worker exited during initialization: {process.returncode}")
        time.sleep(0.2)
    raise TimeoutError("NPU worker readiness timed out")


def sample_phase(writer, raw_file, sampler, trial, phase, duration, interval):
    start = time.monotonic()
    time.sleep(interval)
    rows = []
    while time.monotonic() - start < duration:
        row = sampler.sample()
        row.update({
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": time.monotonic() - start,
            "domain": trial["domain"],
            "target_load_pct": trial["level"],
            "round": trial["round"],
            "trial_order": trial["order"],
            "phase": phase,
        })
        if writer.fieldnames is None:
            writer.fieldnames = list(row.keys())
            writer.writeheader()
        writer.writerow(row)
        raw_file.flush()
        rows.append(row)
        time.sleep(interval)
    return rows


class DynamicWriter:
    def __init__(self, handle):
        self.handle = handle
        self.fieldnames = None
        self.writer = None

    def writeheader(self):
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def writerow(self, row):
        self.writer.writerow(row)


def run_trial(writer, raw_file, trial, args, script_dir, log_dir, state_dir):
    stem = f"{trial['domain']}_r{trial['round']}_o{trial['order']}_l{trial['level']}"
    counter_file = state_dir / f"{stem}.counter"
    ready_file = state_dir / f"{stem}.ready"
    stop_file = state_dir / f"{stem}.stop"
    for path in (counter_file, ready_file, stop_file):
        path.unlink(missing_ok=True)
    npu = start_npu(
        script_dir, args.model, counter_file, ready_file, stop_file,
        log_dir / f"npu_{trial['domain']}_r{trial['round']}_o{trial['order']}_l{trial['level']}.log",
    )
    external = ManagedProcess()
    try:
        wait_ready(npu.process, ready_file)
        sampler = Sampler(counter_file)
        sample_phase(writer, raw_file, sampler, trial, "npu_warmup", args.npu_warmup, args.interval)
        pre_rows = sample_phase(writer, raw_file, sampler, trial, "npu_pre", args.baseline_duration, args.interval)

        external = start_external(
            trial["domain"], trial["level"], script_dir,
            log_dir / f"{trial['domain']}_r{trial['round']}_o{trial['order']}_l{trial['level']}.log",
        )
        wait_healthy(external.process, 0.5)
        sample_phase(writer, raw_file, sampler, trial, "load_settle", args.load_settle, args.interval)
        load_rows = sample_phase(writer, raw_file, sampler, trial, "load_on", args.load_duration, args.interval)

        external.stop()
        sample_phase(writer, raw_file, sampler, trial, "load_recovery", args.load_recovery, args.interval)
        post_rows = sample_phase(writer, raw_file, sampler, trial, "npu_post", args.baseline_duration, args.interval)
        wait_healthy(npu.process, 0.1)
        direct_power = sum(row["npu_ipu_power_w"] for row in load_rows) / len(load_rows)
        direct_activity = sum(row["npu_ipu_activity_mean_pct"] for row in load_rows) / len(load_rows)
        work_rate = sum(row["npu_work_per_s"] for row in load_rows) / len(load_rows)
        if direct_power < 0.20 or direct_activity < 20.0 or work_rate <= 0:
            raise RuntimeError(
                f"NPU direct telemetry invalid: power={direct_power:.3f}W, "
                f"activity={direct_activity:.1f}%, work={work_rate:.2f}/s"
            )
    finally:
        external.stop()
        if npu.process and npu.process.poll() is None:
            stop_file.touch()
            try:
                npu.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        npu.stop()
    time.sleep(args.trial_recovery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--levels", default="0,25,50,75,100")
    parser.add_argument("--domains", default="cpu,gpu")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--npu-warmup", type=float, default=4.0)
    parser.add_argument("--baseline-duration", type=float, default=5.0)
    parser.add_argument("--load-settle", type=float, default=3.0)
    parser.add_argument("--load-duration", type=float, default=12.0)
    parser.add_argument("--load-recovery", type=float, default=3.0)
    parser.add_argument("--trial-recovery", type=float, default=3.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise FileNotFoundError(
            f"Missing ResNet152 model: {args.model}. See the archive model index."
        )
    if not Path(NPU_PYTHON).is_file():
        raise FileNotFoundError(
            f"Missing Ryzen AI Python: {NPU_PYTHON}. Set RYZEN_AI_VENV."
        )
    if "gpu" in args.domains.split(",") and not GPU_BINARY.is_file():
        raise FileNotFoundError(
            f"Missing GPU load binary: {GPU_BINARY}. Set INPU_GPU_LOAD_BINARY."
        )

    if not os.access(PACKAGE_ENERGY, os.R_OK):
        raise PermissionError(f"Package RAPL is not readable: {PACKAGE_ENERGY}")
    direct = read_gpu_metrics()
    if direct["gpu_metrics_format"] != 3:
        raise RuntimeError("gpu_metrics v3.0 is required")

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    log_dir = output_dir / "logs"
    state_dir = output_dir / "state"
    for directory in (data_dir, log_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    levels = [int(value) for value in args.levels.split(",")]
    rng = random.Random(args.seed)
    schedule = []
    domains = [value.strip() for value in args.domains.split(",") if value.strip()]
    if any(domain not in {"cpu", "gpu"} for domain in domains):
        raise ValueError(f"Unsupported domains: {domains}")
    for domain in domains:
        for round_index in range(1, args.rounds + 1):
            order = levels[:]
            rng.shuffle(order)
            for order_index, level in enumerate(order, 1):
                schedule.append({"domain": domain, "round": round_index, "order": order_index, "level": level})

    metadata = vars(args) | {
        "started_at": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        "protocol": "cross_platform_v2_same_model_family",
        "schedule": schedule,
        "npu_power_source": "amdgpu gpu_metrics_v3_0 average_ipu_power",
        "npu_frequency_source": "amdgpu gpu_metrics_v3_0 average_ipuclk_frequency and average_mpipu_frequency",
        "npu_interrupt_source": "/proc/interrupts xdna_mailbox vectors for 0000:05:00.1",
        "npu_workload": "ResNet152 INT8 via VitisAIExecutionProvider with CPU boundary-op fallback",
        "gpu_metrics_filter_us": direct["metrics_filter_us"],
        "source_files_untouched": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    raw_path = data_dir / "raw_samples.csv"
    with open(raw_path, "w", newline="") as raw_file:
        writer = DynamicWriter(raw_file)
        for index, trial in enumerate(schedule, 1):
            print(
                f"[{index}/{len(schedule)}] {trial['domain'].upper()} target={trial['level']}% "
                f"round={trial['round']} order={trial['order']}",
                flush=True,
            )
            run_trial(writer, raw_file, trial, args, script_dir, log_dir, state_dir)

    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "completed"
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Completed: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
