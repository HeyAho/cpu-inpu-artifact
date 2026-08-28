# Observation-state characterization

This subset contains the code and models used to characterize CPU-visible
observation states of integrated NPU execution.

## Artifact map

| Experiment role | Included paths |
| --- | --- |
| Controlled cache disturbance after an NPU/ANE inference request | `linux_intel/characterization_cache/`, `linux_amd/characterization_cache/`, `macos_apple_m2/characterization_cache/`, `macos_apple_m4/characterization_cache/`, `macos_apple_m5pro/characterization_cache/`, `shared/paper_figures/cache_profiles/`, `shared/paper_figures/slc_nfr/` |
| Input-dependent NPU/ANE power observation with all-zero/all-one ResNet50 inputs | `linux_intel/characterization_power/`, `linux_amd/characterization_power/`, `macos_apple_m2/characterization_power/`, `macos_apple_m4/characterization_power/`, `macos_apple_m5pro/characterization_power/`, `shared/characterization_input_power/`, `shared/paper_figures/characterization/` |
| CPU/GPU load impact on reported NPU/ANE power | `macos_apple_m5pro/characterization_cross_component/`, `shared/characterization_cpu_gpu_load/`, `shared/paper_figures/characterization/` |
| CPU-visible interrupt-rate characterization | `linux_intel/characterization_cross_component_interrupt/`, `linux_amd/characterization_cross_component_interrupt/`, `macos_apple_m2/characterization_cross_component_interrupt/`, `macos_apple_m4/characterization_cross_component_interrupt/`, `macos_apple_m5pro/characterization_cross_component/`, `shared/characterization_interrupt/` |

## Model and workload notes

`models/MODEL_INDEX.tsv` lists the model dependencies for these experiments. The required
cache, input-power, and interrupt/cross-component workloads are bundled under
`models/by_platform/`, including Linux ONNX/OpenVINO/VitisAI models and Apple
Core ML packages.
