#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def load_round(path, round_name):
    summary = pd.read_csv(path / "black_white_summary.csv")
    summary = summary[summary["metric"] == "npu_power_w"].copy()
    thresholds = pd.read_csv(path / "black_white_best_threshold_accuracy.csv")
    thresholds = thresholds[["platform", "best_threshold_acc"]]
    frame = summary.merge(thresholds, on="platform")
    frame.insert(0, "round", round_name)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = pd.concat(
        [load_round(args.round1, "round1"), load_round(args.round2, "round2")],
        ignore_index=True,
    )
    columns = [
        "round",
        "platform",
        "black_mean",
        "black_std",
        "white_mean",
        "white_std",
        "delta",
        "direction",
        "pairwise_ordering_errors",
        "pairs",
        "best_threshold_acc",
        "black_n",
        "white_n",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison[columns].to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
