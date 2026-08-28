#!/usr/bin/env python3
import ctypes
from pathlib import Path


GPU_METRICS_PATH = Path("/sys/class/drm/card1/device/gpu_metrics")


class MetricsHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("structure_size", ctypes.c_uint16),
        ("format_revision", ctypes.c_uint8),
        ("content_revision", ctypes.c_uint8),
    ]


class GPUMetricsV30(ctypes.LittleEndianStructure):
    _fields_ = [
        ("common_header", MetricsHeader),
        ("temperature_gfx", ctypes.c_uint16),
        ("temperature_soc", ctypes.c_uint16),
        ("temperature_core", ctypes.c_uint16 * 16),
        ("temperature_skin", ctypes.c_uint16),
        ("average_gfx_activity", ctypes.c_uint16),
        ("average_vcn_activity", ctypes.c_uint16),
        ("average_ipu_activity", ctypes.c_uint16 * 8),
        ("average_core_c0_activity", ctypes.c_uint16 * 16),
        ("average_dram_reads", ctypes.c_uint16),
        ("average_dram_writes", ctypes.c_uint16),
        ("average_ipu_reads", ctypes.c_uint16),
        ("average_ipu_writes", ctypes.c_uint16),
        ("system_clock_counter", ctypes.c_uint64),
        ("average_socket_power", ctypes.c_uint32),
        ("average_ipu_power", ctypes.c_uint16),
        ("average_apu_power", ctypes.c_uint32),
        ("average_gfx_power", ctypes.c_uint32),
        ("average_dgpu_power", ctypes.c_uint32),
        ("average_all_core_power", ctypes.c_uint32),
        ("average_core_power", ctypes.c_uint16 * 16),
        ("average_sys_power", ctypes.c_uint16),
        ("stapm_power_limit", ctypes.c_uint16),
        ("current_stapm_power_limit", ctypes.c_uint16),
        ("average_gfxclk_frequency", ctypes.c_uint16),
        ("average_socclk_frequency", ctypes.c_uint16),
        ("average_vpeclk_frequency", ctypes.c_uint16),
        ("average_ipuclk_frequency", ctypes.c_uint16),
        ("average_fclk_frequency", ctypes.c_uint16),
        ("average_vclk_frequency", ctypes.c_uint16),
        ("average_uclk_frequency", ctypes.c_uint16),
        ("average_mpipu_frequency", ctypes.c_uint16),
        ("current_coreclk", ctypes.c_uint16 * 16),
        ("current_core_maxfreq", ctypes.c_uint16),
        ("current_gfx_maxfreq", ctypes.c_uint16),
        ("throttle_residency_prochot", ctypes.c_uint32),
        ("throttle_residency_spl", ctypes.c_uint32),
        ("throttle_residency_fppt", ctypes.c_uint32),
        ("throttle_residency_sppt", ctypes.c_uint32),
        ("throttle_residency_thm_core", ctypes.c_uint32),
        ("throttle_residency_thm_gfx", ctypes.c_uint32),
        ("throttle_residency_thm_soc", ctypes.c_uint32),
        ("time_filter_alphavalue", ctypes.c_uint32),
    ]


def read_gpu_metrics(path=GPU_METRICS_PATH):
    expected_size = ctypes.sizeof(GPUMetricsV30)
    raw = Path(path).read_bytes()[:expected_size]
    if len(raw) != expected_size:
        raise RuntimeError(f"gpu_metrics read returned {len(raw)} bytes, expected {expected_size}")
    metrics = GPUMetricsV30.from_buffer_copy(raw)
    header = metrics.common_header
    if (
        header.structure_size != expected_size
        or header.format_revision != 3
        or header.content_revision != 0
    ):
        raise RuntimeError(
            "Unsupported gpu_metrics version: "
            f"size={header.structure_size}, format={header.format_revision}, "
            f"content={header.content_revision}"
        )
    ipu_activity = list(metrics.average_ipu_activity)
    core_activity = list(metrics.average_core_c0_activity)
    return {
        "gpu_metrics_format": header.format_revision,
        "gpu_metrics_content": header.content_revision,
        "npu_ipu_power_w": metrics.average_ipu_power / 1000.0,
        "npu_ipuclk_mhz": float(metrics.average_ipuclk_frequency),
        "npu_mpipu_mhz": float(metrics.average_mpipu_frequency),
        "npu_ipu_activity_mean_pct": sum(ipu_activity) / len(ipu_activity),
        "npu_ipu_activity_min_pct": min(ipu_activity),
        "npu_ipu_activity_max_pct": max(ipu_activity),
        "npu_ipu_reads_mb_s": float(metrics.average_ipu_reads),
        "npu_ipu_writes_mb_s": float(metrics.average_ipu_writes),
        "socket_power_w": metrics.average_socket_power / 1000.0,
        "apu_power_w": metrics.average_apu_power / 1000.0,
        "gfx_power_w": metrics.average_gfx_power / 1000.0,
        "all_core_power_w": metrics.average_all_core_power / 1000.0,
        "gpu_activity_pct": float(metrics.average_gfx_activity),
        "gpu_clock_mhz": float(metrics.average_gfxclk_frequency),
        "cpu_c0_activity_mean_pct": sum(core_activity) / len(core_activity),
        "temperature_soc_c": metrics.temperature_soc / 100.0,
        "temperature_gfx_c": metrics.temperature_gfx / 100.0,
        "dram_reads_mb_s": float(metrics.average_dram_reads),
        "dram_writes_mb_s": float(metrics.average_dram_writes),
        "metrics_filter_us": metrics.time_filter_alphavalue,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(read_gpu_metrics(), indent=2))
