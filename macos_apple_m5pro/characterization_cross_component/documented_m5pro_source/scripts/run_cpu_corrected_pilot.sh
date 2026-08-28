#!/bin/zsh
set -euo pipefail

ROOT="external/m5_tasks/m5pro_migrated_from_m2_cpu_gpu_ane_impact_20260805"
PYTHON="$ROOT/.venv/bin/python"
MODEL="external/combined_m5/models/resnet152_PyTorch.mlmodelc"
MONITOR="$ROOT/bin/ANEPowerMonitor_M5Pro_CPU_domains"

exec "$PYTHON" "$ROOT/scripts/run_experiment.py" \
  --output-dir "$ROOT/pilot_cpu_corrected" \
  --platform apple_m5_pro \
  --model "$MODEL" \
  --monitor "$MONITOR" \
  --cpu-workers 18 \
  --cpu-matrix-size 512 \
  --load-cycle-ms 200 \
  --domains cpu \
  --levels 0,100 \
  --rounds 1 \
  --ane-warmup 2 \
  --baseline-duration 2 \
  --load-settle 2 \
  --load-duration 5 \
  --load-recovery 2 \
  --trial-recovery 2 \
  --interval 0.05 \
  --seed 20260713
