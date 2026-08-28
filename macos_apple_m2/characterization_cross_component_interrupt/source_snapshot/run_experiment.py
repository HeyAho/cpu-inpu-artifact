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

import psutil


DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[3]
    / "models/by_platform/macos_apple_m2/characterization_cross_component_interrupt"
    / "resnet152_PyTorch.mlmodelc"
)
DEFAULT_MONITOR = "external/ANEPowerMonitor"


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

    def stop(self, timeout=10):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        if self.log_handle:
            self.log_handle.close()


def launch(command, log_path, cwd=None):
    log_handle = open(log_path, "a")
    process = subprocess.Popen(
        command, cwd=cwd, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedProcess(process, log_handle)


def wait_ready(process, ready_path=None, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path is not None and ready_path.exists():
            return
        if ready_path is None and process.poll() is None:
            time.sleep(1.0)
            if process.poll() is None:
                return
        if process.poll() is not None:
            raise RuntimeError(f"Process exited during initialization: {process.returncode}")
        time.sleep(0.2)
    raise TimeoutError("Process readiness timeout")


def phase_sample(writer, trial, phase, duration, counter_file, interval):
    start_epoch = time.time()
    start_mono = time.monotonic()
    previous_time = start_mono
    previous_count = read_counter(counter_file)
    psutil.cpu_percent(interval=None)
    rows = []
    time.sleep(interval)
    while time.monotonic() - start_mono < duration:
        now_mono = time.monotonic()
        now_count = read_counter(counter_file)
        elapsed = now_mono - previous_time
        row = {
            "timestamp": time.time(),
            "domain": trial["domain"],
            "target_load_pct": trial["level"],
            "round": trial["round"],
            "trial_order": trial["order"],
            "phase": phase,
            "cpu_total_util_pct": psutil.cpu_percent(interval=None),
            "ane_infer_per_s": max(0, now_count - previous_count) / elapsed,
            "ane_infer_count": now_count,
        }
        writer.writerow(row)
        rows.append(row)
        previous_time = now_mono
        previous_count = now_count
        time.sleep(interval)
    return start_epoch, time.time(), rows


def monitor_power_mean(path, start_epoch, end_epoch):
    with open(path) as handle:
        rows = [row for row in csv.DictReader(handle) if start_epoch <= float(row["Timestamp"]) < end_epoch]
    if not rows:
        return 0.0
    return sum(float(row.get("ANE", 0)) for row in rows) / len(rows)


def start_external(trial, script_dir, log_path):
    if trial["level"] == 0:
        return ManagedProcess()
    script = "cpu_load_worker.py" if trial["domain"] == "cpu" else "gpu_load_worker.py"
    return launch([
        sys.executable, str(script_dir / script), "--duty-cycle", str(trial["level"]),
    ], log_path)


def run_trial(trial, args, output_dir, script_dir, system_writer, phase_writer):
    stem = f"{trial['domain']}_r{trial['round']}_o{trial['order']}_l{trial['level']}"
    trial_dir = output_dir / "data" / "trials" / stem
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    state_dir = output_dir / "state"
    counter_file = state_dir / f"{stem}.counter"
    ready_file = state_dir / f"{stem}.ready"
    stop_file = state_dir / f"{stem}.stop"
    for path in (counter_file, ready_file, stop_file):
        path.unlink(missing_ok=True)
    monitor = launch([args.monitor, "monitor.csv"], log_dir / f"monitor_{stem}.log", cwd=trial_dir)
    ane = ManagedProcess()
    external = ManagedProcess()
    try:
        wait_ready(monitor.process)
        ane = launch([
            sys.executable, str(script_dir / "ane_worker.py"),
            "--model", args.model, "--counter-file", str(counter_file),
            "--ready-file", str(ready_file), "--stop-file", str(stop_file),
            "--target-rate", str(args.ane_target_rate),
        ], log_dir / f"ane_{stem}.log")
        wait_ready(ane.process, ready_file)
        phases = [
            ("ane_warmup", args.ane_warmup),
            ("ane_pre", args.baseline_duration),
        ]
        phase_records = []
        for phase, duration in phases:
            start, end, rows = phase_sample(system_writer, trial, phase, duration, counter_file, args.interval)
            phase_records.append((phase, start, end))
        external = start_external(trial, script_dir, log_dir / f"load_{stem}.log")
        if external.process:
            wait_ready(external.process)
        for phase, duration in (("load_settle", args.load_settle), ("load_on", args.load_duration)):
            start, end, rows = phase_sample(system_writer, trial, phase, duration, counter_file, args.interval)
            phase_records.append((phase, start, end))
            if phase == "load_on":
                load_rows = rows
                load_start, load_end = start, end
        external.stop()
        for phase, duration in (("load_recovery", args.load_recovery), ("ane_post", args.baseline_duration)):
            start, end, rows = phase_sample(system_writer, trial, phase, duration, counter_file, args.interval)
            phase_records.append((phase, start, end))
        if ane.process.poll() is not None:
            raise RuntimeError(f"ANE worker exited early: {ane.process.returncode}")
        monitor_csv = trial_dir / "monitor.csv"
        ane_power = monitor_power_mean(monitor_csv, load_start, load_end)
        work_rate = sum(row["ane_infer_per_s"] for row in load_rows) / len(load_rows)
        if ane_power < 0.5 or work_rate <= 0:
            raise RuntimeError(f"ANE validation failed: power={ane_power:.3f}W work={work_rate:.2f}/s")
        for phase, start, end in phase_records:
            phase_writer.writerow({
                "trial": stem, "domain": trial["domain"], "target_load_pct": trial["level"],
                "round": trial["round"], "trial_order": trial["order"], "phase": phase,
                "start_epoch": start, "end_epoch": end,
            })
    finally:
        external.stop()
        if ane.process and ane.process.poll() is None:
            stop_file.touch()
            try:
                ane.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        ane.stop()
        monitor.stop()
    time.sleep(args.trial_recovery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument("--ane-target-rate", type=float, default=100.0)
    parser.add_argument("--domains", default="cpu,gpu")
    parser.add_argument("--levels", default="0,25,50,75,100")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ane-warmup", type=float, default=4.0)
    parser.add_argument("--baseline-duration", type=float, default=5.0)
    parser.add_argument("--load-settle", type=float, default=3.0)
    parser.add_argument("--load-duration", type=float, default=12.0)
    parser.add_argument("--load-recovery", type=float, default=3.0)
    parser.add_argument("--trial-recovery", type=float, default=3.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for directory in (output_dir / "data", output_dir / "logs", output_dir / "state"):
        directory.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    levels = [int(value) for value in args.levels.split(",")]
    rng = random.Random(args.seed)
    schedule = []
    for domain in args.domains.split(","):
        for round_index in range(1, args.rounds + 1):
            order = levels[:]
            rng.shuffle(order)
            for order_index, level in enumerate(order, 1):
                schedule.append({"domain": domain, "round": round_index, "order": order_index, "level": level})
    metadata = vars(args) | {
        "hostname": os.uname().nodename, "started_at": datetime.now().isoformat(),
        "protocol": "cross_platform_ane_v1", "schedule": schedule,
        "ane_power_source": "IOReport Energy Model ANE channel",
        "gpu_source": "IOReport GPU and GPU Performance States",
        "privilege": "ordinary user; no sudo",
        "ane_frequency_available": False, "ane_interrupt_trace_available": False,
        "source_files_untouched": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    system_path = output_dir / "data" / "system_samples.csv"
    phase_path = output_dir / "data" / "phase_intervals.csv"
    with system_path.open("w", newline="") as system_handle, phase_path.open("w", newline="") as phase_handle:
        system_fields = ["timestamp", "domain", "target_load_pct", "round", "trial_order", "phase", "cpu_total_util_pct", "ane_infer_per_s", "ane_infer_count"]
        phase_fields = ["trial", "domain", "target_load_pct", "round", "trial_order", "phase", "start_epoch", "end_epoch"]
        system_writer = csv.DictWriter(system_handle, fieldnames=system_fields)
        phase_writer = csv.DictWriter(phase_handle, fieldnames=phase_fields)
        system_writer.writeheader(); phase_writer.writeheader()
        for index, trial in enumerate(schedule, 1):
            print(f"[{index}/{len(schedule)}] {trial['domain'].upper()} target={trial['level']}% round={trial['round']} order={trial['order']}", flush=True)
            run_trial(trial, args, output_dir, script_dir, system_writer, phase_writer)
            system_handle.flush(); phase_handle.flush()
    metadata["completed_at"] = datetime.now().isoformat(); metadata["status"] = "completed"
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
