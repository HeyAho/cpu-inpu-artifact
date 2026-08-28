#!/bin/zsh
set -euo pipefail

ROOT="external/m5_tasks/m5pro_migrated_from_m2_cpu_gpu_ane_impact_20260805"
PYTHON="$ROOT/.venv/bin/python"
MODEL="external/combined_m5/models/resnet152_PyTorch.mlmodelc"
MONITOR="$ROOT/bin/ANEPowerMonitor_M5Pro_CPU_domains"
OUTPUT="$ROOT/data_gpu_2048_latest_rerun"

exec "$PYTHON" "$ROOT/scripts/run_experiment.py" \
  --output-dir "$OUTPUT" \
  --platform apple_m5_pro \
  --model "$MODEL" \
  --monitor "$MONITOR" \
  --ane-target-rate 0 \
  --cpu-workers 18 \
  --cpu-matrix-size 512 \
  --gpu-matrix-size 2048 \
  --load-cycle-ms 200 \
  --domains gpu \
  --levels 0,25,50,75,100 \
  --rounds 3 \
  --ane-warmup 5 \
  --baseline-duration 8 \
  --load-settle 8 \
  --load-duration 20 \
  --load-recovery 8 \
  --trial-recovery 8 \
  --interval 0.05 \
  --seed 20260713
