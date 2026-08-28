#!/usr/bin/env python3
"""Experiment 04: B=1 ResNet50 black/white ANE side-channel collection."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, time
from pathlib import Path
import numpy as np
import coremltools as ct


def terminate_monitor(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], np.empty((0, 0))
    header = rows[0]
    numeric = []
    for row in rows[1:]:
        try:
            numeric.append([float(v) for v in row])
        except (ValueError, TypeError):
            continue
    return header, np.asarray(numeric, dtype=float)


def run_one(model, input_data: np.ndarray, monitor: str, csv_path: Path,
            baseline: float, active: float, cooldown: float) -> dict:
    csv_path.unlink(missing_ok=True)
    probe = subprocess.Popen([monitor, str(csv_path)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    # Let the monitor establish its IOReport subscription before baseline.
    time.sleep(2.0)
    time.sleep(baseline)
    started = time.monotonic()
    count = 0
    while time.monotonic() - started < active:
        model.predict({"input_1": input_data})
        count += 1
    elapsed = time.monotonic() - started
    time.sleep(cooldown)
    terminate_monitor(probe)
    if not csv_path.exists():
        raise RuntimeError(f"monitor did not create {csv_path}")
    header, arr = read_csv(csv_path)
    if arr.shape[0] < 5 or "ANE" not in header:
        raise RuntimeError(f"invalid monitor output {csv_path}: shape={arr.shape}, header={header}")
    idx = {name: i for i, name in enumerate(header)}
    t = arr[:, idx["Timestamp"]] - arr[0, idx["Timestamp"]]
    base = t < baseline
    on = (t >= baseline) & (t <= baseline + active)
    if not np.any(on):
        raise RuntimeError(f"no active samples in {csv_path}; elapsed range={t[-1]:.2f}s")
    ane = arr[:, idx["ANE"]]
    def vals(mask, name):
        return arr[mask, idx[name]] if name in idx and np.any(mask) else np.array([])
    gpu_on = vals(on, "GPU_Freq")
    result = {
        "baseline_ane": float(np.mean(ane[base])) if np.any(base) else 0.0,
        "active_ane": float(np.mean(ane[on])),
        "active_ane_median": float(np.median(ane[on])),
        "active_ane_std": float(np.std(ane[on])),
        "active_ane_p90": float(np.percentile(ane[on], 90)),
        "active_ane_peak": float(np.max(ane[on])),
        "delta_ane": float(np.mean(ane[on]) - np.mean(ane[base])) if np.any(base) else float(np.mean(ane[on])),
        "baseline_samples": int(np.sum(base)),
        "active_samples": int(np.sum(on)),
        "monitor_samples": int(arr.shape[0]),
        "monitor_duration_s": float(t[-1]),
        "inference_seconds": float(elapsed),
        "throughput": float(count / elapsed),
    }
    if gpu_on.size:
        result["active_gpu_freq_mean"] = float(np.mean(gpu_on))
        result["active_gpu_freq_max"] = float(np.max(gpu_on))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../../models/by_platform/macos_apple_m5pro/characterization_power/ResNet50_B1_Size224_ANE.mlpackage")
    ap.add_argument("--monitor", default="../../../external/macos/ANEPowerMonitor")
    ap.add_argument("--output", default="data")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--active", type=float, default=15.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--between", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true", help="one 3-second run per class")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    black = np.zeros((1, 3, 224, 224), dtype=np.float32)
    white = np.ones((1, 3, 224, 224), dtype=np.float32)
    # Force model compilation and confirm the input interface before collection.
    model.predict({"input_1": black})
    reps = 1 if args.smoke else args.reps
    active = 3.0 if args.smoke else args.active
    all_results = {"black": [], "white": []}
    for i in range(reps):
        for label, data in (("black", black), ("white", white)):
            path = output / f"resnet_power_{label}_r{i}.csv"
            print(f"[{label}] rep {i + 1}/{reps}", flush=True)
            result = run_one(model, data, args.monitor, path, args.baseline, active, args.cooldown)
            all_results[label].append(result)
            print(f"  ANE baseline={result['baseline_ane']:.3f} active={result['active_ane']:.3f} "
                  f"delta={result['delta_ane']:+.3f} peak={result['active_ane_peak']:.3f} "
                  f"throughput={result['throughput']:.1f}/s samples={result['active_samples']}", flush=True)
            if not args.smoke:
                time.sleep(args.between)
    metadata = {
        "experiment": "04_npu_power_black_white",
        "platform": "M5 Pro",
        "model": str(model_path),
        "compute_units": "CPU_AND_NE",
        "batch_size": 1,
        "input_shape": [1, 3, 224, 224],
        "black_input": "zeros",
        "white_input": "ones",
        "repetitions_per_class": reps,
        "baseline_seconds": args.baseline,
        "active_seconds": active,
        "cooldown_seconds": args.cooldown,
        "monitor": args.monitor,
        "results": all_results,
    }
    (output / "collection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output / 'collection_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
