#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import random
import re
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PMT_GUID = "0x130670b2"
PMT_ENERGY_OFFSET = 0x628
PMT_WORKPOINT_OFFSET = 0x68
ENERGY_SCALE = 16384.0
NPU_DRIVER = "/sys/bus/pci/drivers/intel_vpu"
NPU_PCI_ID = "0000:00:0b.0"
RAPL_PACKAGE = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
GPU_FREQ_PATHS = (
    "/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz",
    "/sys/class/drm/card1/gt_act_freq_mhz",
)


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


def find_npu_sysfs():
    for path in glob.glob(f"{NPU_DRIVER}/0000:*"):
        if os.path.isdir(path):
            return Path(path)
    raise RuntimeError("Intel NPU sysfs directory not found")


def find_npu_telem():
    for path in sorted(glob.glob("/sys/class/intel_pmt/telem*")):
        if read_text(Path(path) / "guid") == PMT_GUID:
            telem = Path(path) / "telem"
            with telem.open("rb") as handle:
                handle.read(8)
            return telem
    raise RuntimeError("Readable Intel NPU PMT telemetry not found")


def read_cpu_ticks():
    values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def read_gpu_frequency():
    for path in GPU_FREQ_PATHS:
        value = read_text(path)
        if value is not None:
            return float(value)
    return 0.0


def read_gpu_engine_ns(pid):
    if pid is None:
        return None
    total = 0
    found = False
    for fdinfo in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            content = Path(fdinfo).read_text()
        except OSError:
            continue
        match = re.search(r"^drm-engine-compute:\s+(\d+)\s+ns$", content, re.MULTILINE)
        if match:
            total += int(match.group(1))
            found = True
    return total if found else None


def read_npu_irq_counts():
    counts = {}
    cpu_count = os.cpu_count() or 1
    for line in Path("/proc/interrupts").read_text().splitlines():
        if NPU_PCI_ID not in line or "intel_vpu" not in line:
            continue
        fields = line.split()
        irq = fields[0].rstrip(":")
        values = [int(value) for value in fields[1:1 + cpu_count] if value.isdigit()]
        if values:
            counts[irq] = sum(values)
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
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        if self.log_handle:
            self.log_handle.close()


def launch(command, log_path):
    log_handle = open(log_path, "a")
    process = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    return ManagedProcess(process, log_handle)


def start_npu(script_dir, model, counter_file, ready_file, log_path):
    return launch([
        "taskset", "-c", "0,1", sys.executable,
        str(script_dir / "npu_infer_worker.py"),
        "--model", model,
        "--counter-file", str(counter_file),
        "--ready-file", str(ready_file),
    ], log_path)


def start_external(domain, level, script_dir, log_path):
    if level == 0:
        return ManagedProcess()
    if domain == "cpu":
        command = [sys.executable, str(script_dir / "cpu_load_worker.py"), "--duty-cycle", str(level)]
    else:
        command = [sys.executable, str(script_dir / "gpu_load_worker.py"), "--duty-cycle", str(level)]
    return launch(command, log_path)


class Sampler:
    def __init__(self, counter_file, gpu_pid=None):
        self.counter_file = counter_file
        self.gpu_pid = gpu_pid
        self.npu_sysfs = find_npu_sysfs()
        self.telem = find_npu_telem()
        self.package_max = read_int(RAPL_MAX)
        self.last_irq_counts = read_npu_irq_counts()
        self.previous = self._snapshot()

    def _snapshot(self):
        now = time.monotonic()
        with self.telem.open("rb") as handle:
            buffer = handle.read()
        energy_j = struct.unpack_from("<Q", buffer, PMT_ENERGY_OFFSET)[0] / ENERGY_SCALE
        workpoint = struct.unpack_from("<I", buffer, PMT_WORKPOINT_OFFSET)[0]
        total, idle = read_cpu_ticks()
        package_uj = read_int(RAPL_PACKAGE, -1) if os.access(RAPL_PACKAGE, os.R_OK) else -1
        return {
            "time": now,
            "energy_j": energy_j,
            "npu_freq": (workpoint & 0xFF) * (100.0 / 3.0),
            "busy_us": read_int(self.npu_sysfs / "npu_busy_time_us"),
            "cpu_total": total,
            "cpu_idle": idle,
            "package_uj": package_uj,
            "gpu_ns": read_gpu_engine_ns(self.gpu_pid),
            "gpu_freq": read_gpu_frequency(),
            "work_count": read_counter(self.counter_file),
        }

    def _irq_rate(self, elapsed):
        current = read_npu_irq_counts()
        delta = 0
        for irq, count in current.items():
            if irq in self.last_irq_counts and count >= self.last_irq_counts[irq]:
                delta += count - self.last_irq_counts[irq]
            self.last_irq_counts[irq] = count
        return delta / elapsed, len(current)

    def sample(self):
        current = self._snapshot()
        previous = self.previous
        self.previous = current
        elapsed = current["time"] - previous["time"]
        cpu_delta = current["cpu_total"] - previous["cpu_total"]
        idle_delta = current["cpu_idle"] - previous["cpu_idle"]
        gpu_busy = 0.0
        if current["gpu_ns"] is not None and previous["gpu_ns"] is not None:
            gpu_busy = min(100.0, max(0.0, (current["gpu_ns"] - previous["gpu_ns"]) / (elapsed * 1e9) * 100.0))
        package_power = None
        if current["package_uj"] >= 0 and previous["package_uj"] >= 0:
            delta = current["package_uj"] - previous["package_uj"]
            if delta < 0 and self.package_max > 0:
                delta += self.package_max
            package_power = max(0, delta) / 1e6 / elapsed
        irq_rate, irq_vectors = self._irq_rate(elapsed)
        return {
            "platform": "intel_meteor_lake",
            "workload": "openvino_resnet152_npu",
            "npu_power_w": max(0.0, current["energy_j"] - previous["energy_j"]) / elapsed,
            "npu_primary_freq_mhz": current["npu_freq"],
            "npu_secondary_freq_mhz": None,
            "npu_activity_pct": min(100.0, max(0.0, (current["busy_us"] - previous["busy_us"]) / (elapsed * 1e6) * 100.0)),
            "npu_work_per_s": max(0, current["work_count"] - previous["work_count"]) / elapsed,
            "npu_irq_per_s": irq_rate,
            "npu_irq_vectors": irq_vectors,
            "cpu_total_util_pct": 100.0 * (1.0 - idle_delta / cpu_delta) if cpu_delta > 0 else 0.0,
            "gpu_activity_pct": gpu_busy,
            "gpu_freq_mhz": current["gpu_freq"],
            "package_power_w": package_power,
        }


class DynamicWriter:
    def __init__(self, handle):
        self.handle = handle
        self.fieldnames = None
        self.writer = None

    def writerow(self, row):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(row)


def wait_process(process, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            raise RuntimeError(f"Workload exited early with code {process.returncode}")
        time.sleep(0.2)


def wait_ready(process, ready_file, timeout=90):
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
        writer.writerow(row)
        raw_file.flush()
        rows.append(row)
        time.sleep(interval)
    return rows


def run_trial(writer, raw_file, trial, args, script_dir, log_dir, state_dir):
    stem = f"{trial['domain']}_r{trial['round']}_o{trial['order']}_l{trial['level']}"
    counter_file = state_dir / f"{stem}.counter"
    ready_file = state_dir / f"{stem}.ready"
    for path in (counter_file, ready_file):
        path.unlink(missing_ok=True)
    npu = start_npu(script_dir, args.model, counter_file, ready_file, log_dir / f"npu_{stem}.log")
    external = ManagedProcess()
    try:
        wait_ready(npu.process, ready_file)
        sampler = Sampler(counter_file)
        sample_phase(writer, raw_file, sampler, trial, "npu_warmup", args.npu_warmup, args.interval)
        sample_phase(writer, raw_file, sampler, trial, "npu_pre", args.baseline_duration, args.interval)
        external = start_external(trial["domain"], trial["level"], script_dir, log_dir / f"{stem}.log")
        wait_process(external.process, 0.5)
        sampler.gpu_pid = external.process.pid if trial["domain"] == "gpu" and external.process else None
        sampler.previous = sampler._snapshot()
        sample_phase(writer, raw_file, sampler, trial, "load_settle", args.load_settle, args.interval)
        load_rows = sample_phase(writer, raw_file, sampler, trial, "load_on", args.load_duration, args.interval)
        external.stop()
        sampler.gpu_pid = None
        sampler.previous = sampler._snapshot()
        sample_phase(writer, raw_file, sampler, trial, "load_recovery", args.load_recovery, args.interval)
        sample_phase(writer, raw_file, sampler, trial, "npu_post", args.baseline_duration, args.interval)
        power = sum(row["npu_power_w"] for row in load_rows) / len(load_rows)
        activity = sum(row["npu_activity_pct"] for row in load_rows) / len(load_rows)
        work = sum(row["npu_work_per_s"] for row in load_rows) / len(load_rows)
        if power < 0.05 or activity < 1.0 or work <= 0:
            raise RuntimeError(f"NPU telemetry invalid: power={power:.3f} activity={activity:.1f} work={work:.2f}")
    finally:
        external.stop()
        npu.stop()
    time.sleep(args.trial_recovery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parents[3]
            / "models/by_platform/linux_intel/characterization_cross_component_interrupt"
            / "resnet152_ov_dynamic.xml"
        ),
    )
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
    if not os.access(find_npu_telem(), os.R_OK):
        raise PermissionError("Intel NPU PMT telemetry is not readable")
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    log_dir = output_dir / "logs"
    state_dir = output_dir / "state"
    for directory in (data_dir, log_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    levels = [int(value) for value in args.levels.split(",")]
    domains = [value.strip() for value in args.domains.split(",")]
    rng = random.Random(args.seed)
    schedule = []
    for domain in domains:
        for round_index in range(1, args.rounds + 1):
            order = levels[:]
            rng.shuffle(order)
            for order_index, level in enumerate(order, 1):
                schedule.append({"domain": domain, "round": round_index, "order": order_index, "level": level})
    metadata = vars(args) | {
        "started_at": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        "protocol": "cross_platform_v1",
        "schedule": schedule,
        "npu_power_source": "Intel PMT VPU_ENERGY delta",
        "npu_frequency_source": "Intel PMT VPU workpoint",
        "npu_interrupt_source": "/proc/interrupts intel_vpu vector for 0000:00:0b.0",
        "package_power_available": os.access(RAPL_PACKAGE, os.R_OK),
        "source_files_untouched": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    raw_path = data_dir / "raw_samples.csv"
    with raw_path.open("w", newline="") as raw_file:
        writer = DynamicWriter(raw_file)
        for index, trial in enumerate(schedule, 1):
            print(f"[{index}/{len(schedule)}] {trial['domain'].upper()} target={trial['level']}% round={trial['round']} order={trial['order']}", flush=True)
            run_trial(writer, raw_file, trial, args, script_dir, log_dir, state_dir)
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "completed"
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Completed: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
