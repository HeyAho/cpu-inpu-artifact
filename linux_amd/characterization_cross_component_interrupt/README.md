# AMD XDNA2 cross-component interrupt characterization

This is the AMD source used by the paper's aligned ResNet152 experiment. It
measures `xdna_mailbox` vectors for PCI device `0000:05:00.1` from
`/proc/interrupts`, while the NPU worker runs ResNet152 through the
`VitisAIExecutionProvider`. NPU power, frequency, and activity are read from
amdgpu `gpu_metrics` v3.0.

Place `resnet152_int8.onnx` in
`../../models/by_platform/linux_amd/characterization_cross_component_interrupt/`,
create the Ryzen AI environment at `.venv/` or set `RYZEN_AI_VENV`, and set
`INPU_GPU_LOAD_BINARY` when GPU-load trials are requested.

```bash
python3 source_snapshot/run_experiment.py \
  --output-dir results/raw --rounds 3
python3 source_snapshot/analyze_results.py \
  --experiment-dir results/raw
```
