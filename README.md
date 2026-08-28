# CPU-iNPU observation-state artifact

This directory contains the anonymized code and models used to characterize
CPU-visible observation states of integrated NPU execution, including cache
state, power/operating-state telemetry, and interrupt activity.

Included contents:

- source code
- scripts
- README files
- configuration and launch files
- the model/workload files listed as `bundled` in
  `models/MODEL_INDEX.tsv`

## Layout

- `linux_intel/`: observation-state code for Intel Core Ultra 7 155H
- `linux_amd/`: observation-state code for AMD Ryzen AI 7 H 350
- `macos_apple_m2/`: observation-state code for Apple M2
- `macos_apple_m4/`: observation-state code for Apple M4
- `macos_apple_m5pro/`: observation-state code for Apple M5 Pro
- `models/`: model/workload inventory and bundled model files

## Anonymous path policy

Absolute personal paths were rewritten to relative placeholders.
The main placeholders are:

- `external/`: local copies of external tools, runtime roots, and model
  dependencies
- `assets/`: recovered input assets or source snapshots
- `figures/`: manuscript figure targets
- `home/`: generic user-home placeholders
- `tools/`: helper scripts shipped with this anonymous bundle

## How to use

1. Restore any required external runtime or monitor dependency under
   the matching placeholder directory.
2. Edit a script only if it still points to a local placeholder you do not
   provide.
3. Run the platform-specific README files in the corresponding subdirectories.

## Verification scope

This copy preserves the source layout for observation-state experiments:
cache disturbance, input-dependent NPU/ANE power, CPU/GPU load effects on
NPU/ANE power, and iNPU/ANE interrupt-rate characterization.
