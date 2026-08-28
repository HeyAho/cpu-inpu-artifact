# Runtime model bundles

`MODEL_INDEX.tsv` is the authoritative platform/experiment inventory. Model
assets are organized under `by_platform/<platform>/<experiment>/`. Entries
marked `bundled` are included directly in this artifact.

This directory contains the model and workload packages required by the
observation-state characterization experiments.

## Cache characterization

The Linux platforms include `heavy_eviction_exact_xint8.onnx`. The Apple
platforms include `EvictionModel_Heavy.mlpackage`.

## Input-dependent power

The `by_platform/<platform>/characterization_power/` directories contain the
ResNet50 models used by the documented black-vs-white collection. The Linux
platforms include ONNX variants for OpenVINO and VitisAI execution. The M2 and
M4 use `ResNet50_ANE.mlpackage`; the M5 Pro uses
`ResNet50_B1_Size224_ANE.mlpackage`. These workloads are evaluated with all-zero
and all-one tensors.

The M5 Pro package is retained separately because its recorded conversion has
a different Core ML package hash, even though it implements the same ResNet50
topology and input protocol.  This preserves the exact per-platform runtime
provenance rather than silently substituting weights.

## Interrupt and cross-component characterization

The Linux platforms include ResNet152 workloads for OpenVINO and VitisAI. The
Apple platforms include the Core ML workloads used by the corresponding
interrupt or CPU/GPU-load experiments.
