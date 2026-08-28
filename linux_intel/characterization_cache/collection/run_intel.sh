#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIALS="${1:-50}"
N_INFER="${2:-1}"
CPU="${3:--1}"
OUTPUT_DIR="${4:-$ROOT_DIR/results/intel}"

exec "$ROOT_DIR/bin/slc_sharing_test_x86" \
  "$ROOT_DIR/runtime/intel_openvino_server.py" \
  "$ROOT_DIR/../../../models/by_platform/linux_intel/characterization_cache/heavy_eviction_exact_xint8.onnx" \
  "$TRIALS" "$N_INFER" "$CPU" "$OUTPUT_DIR"
