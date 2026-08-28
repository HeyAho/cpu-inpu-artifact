#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--domains", default="cpu,gpu")
    parser.add_argument("--expected-levels", default="0,25,50,75,100")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    intervals = pd.read_csv(root / "data" / "phase_intervals.csv")
    system = pd.read_csv(root / "data" / "system_samples.csv")
    trials = intervals[["trial", "domain", "target_load_pct", "round"]].drop_duplicates()
    domains = tuple(value.strip() for value in args.domains.split(",") if value.strip())
    expected_levels = tuple(int(value) for value in args.expected_levels.split(","))
    expected_trials = args.expected_rounds * len(domains) * len(expected_levels)
    required_phases = {
        "ane_warmup", "ane_pre", "load_settle", "load_on", "load_recovery", "ane_post"
    }

    problems = []
    if len(trials) != expected_trials:
        problems.append(f"trial_count={len(trials)} expected={expected_trials}")
    expected_conditions = {
        (domain, level, round_index)
        for domain in domains
        for level in expected_levels
        for round_index in range(1, args.expected_rounds + 1)
    }
    actual_conditions = set(map(tuple, trials[["domain", "target_load_pct", "round"]].itertuples(index=False, name=None)))
    if actual_conditions != expected_conditions:
        problems.append("condition grid is incomplete or duplicated")

    trial_checks = []
    for trial in sorted(trials.trial):
        phases = set(intervals.loc[intervals.trial == trial, "phase"])
        monitor_path = root / "data" / "trials" / trial / "monitor.csv"
        monitor_rows = 0
        ane_nonzero_rows = 0
        if monitor_path.exists():
            monitor = pd.read_csv(monitor_path)
            monitor_rows = len(monitor)
            ane_nonzero_rows = int((pd.to_numeric(monitor.get("ANE", 0), errors="coerce") > 0).sum())
        trial_system = system[system.trial == trial]
        load_rows = trial_system[trial_system.phase == "load_on"]
        check = {
            "trial": trial,
            "phases_complete": required_phases.issubset(phases),
            "monitor_rows": monitor_rows,
            "ane_nonzero_rows": ane_nonzero_rows,
            "load_on_system_rows": len(load_rows),
            "mean_ane_infer_per_s": float(load_rows.ane_infer_per_s.mean()) if len(load_rows) else 0.0,
        }
        trial_checks.append(check)
        if not check["phases_complete"] or monitor_rows == 0 or ane_nonzero_rows == 0 or check["mean_ane_infer_per_s"] <= 0:
            problems.append(f"invalid trial: {trial}")

    result = {
        "valid": not problems,
        "expected_trials": expected_trials,
        "observed_trials": len(trials),
        "problems": problems,
        "trial_checks": trial_checks,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("valid", "expected_trials", "observed_trials", "problems")}, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
