# M5 Pro 迁移 M2 CPU/GPU 负载影响 ANE 实验

## 目标

在 Apple M5 Pro 上复现 M2 的真实计算负载实验，测量并发 CPU/GPU 负载对饱和 ANE 推理功耗与吞吐的影响。新实验不复用旧 M5/M5 Pro 数据，也不修改 M2 源代码和原始结果。

## 对齐设置

- ANE：CoreML ResNet152，`CPU_AND_NE`，不限速连续推理。
- CPU：Accelerate FP32 SGEMM 512x512，全部 18 核（6 个 Super cores + 12 个 Performance cores），200 ms 占空比周期。
- GPU：最终 ANE Power 实验使用 MPS FP32 矩阵乘 4096x4096，200 ms 占空比周期。
- 负载：0%、25%、50%、75%、100%，CPU/GPU 各 3 轮。
- 时序：5 s warmup、8 s pre、8 s settle、20 s load、8 s recovery、8 s post、8 s trial recovery。
- 采样：系统状态和 ANE 吞吐每 50 ms；IOReport 功耗约每 10--20 ms。
- 权限：普通用户，无 `sudo`。

M5 Pro 上只有经过活动验证的通道才计入结果。CPU 实验报告 ANE 功耗与推理吞吐；最终 GPU 结果只报告 ANE Power。若无法验证稳定的 ANE IRQ 通道，则不以旧芯片的通道名称替代。

## 目录

- `source_m2/`：M2 执行器与协议快照，只读保存。
- `scripts/`：M5 Pro 执行、分析和验证脚本。
- `configs/protocol.json`：机器可读实验协议。
- `data_full/`：首轮 CPU/GPU 联合采集；其中 GPU 结果有效，CPU 功耗只含旧监控器的 `PCPU` 通道。
- `data_cpu_corrected/`：使用 Super/Performance/Total 修正通道重新采集的正式 CPU 数据。
- `data_gpu_4096_ane_power_rerun2/`：最终 GPU ANE Power 原始五档、3 轮重测数据。
- `data_gpu_4096_ane_power_50_replacement/`：50% 异常点的独立重测数据。
- `data_gpu_4096_ane_power_rerun2/results_corrected_50_replacement/`：最终采用的 GPU ANE Power 修正表、图和审计记录。
- `results/`：逐阶段、逐 trial、聚合统计和图。
- `logs/`：pilot、正式采集和分析日志。

## 正式命令

```bash
nohup ./scripts/run_formal.sh > logs/full_run.log 2>&1 &
```

采集完成后运行：

```bash
python3 scripts/validate_experiment.py \
  --experiment-dir data_full --expected-rounds 3 --output validation.json

python3 scripts/analyze_results.py \
  --experiment-dir data_full --platform apple_m5_pro
```

采集使用实验目录内兼容 Core ML 的 Python 3.9 环境；分析使用 M5 Pro
已有的 Miniforge 科学计算环境。两者均不修改系统 Python。

## 完成状态与关键结果

最终采用的 CPU 与 GPU 五档数据各包含 15 个 trial，另有 1 个 50% GPU 独立重测 trial，均通过完整性验证。CPU
修正版直接记录 `CPU_Super`、`CPU_Performance_0/1`、合并后的
`CPU_Performance` 和 `CPU_Total`；首轮联合数据中的 CPU 功耗字段已被修正版取代。

- CPU 100%：ANE power 为同轮基线的 68.25% +/- 21.29%，throughput 为 43.31% +/- 22.55%。
- GPU 100%：最终 ANE Power 为同轮基线的 96.00% +/- 1.09%。
- M5 Pro 的 M2/M4 IOReport IRQ 通道均无法创建订阅，因此未报告 IRQ 结果。

完整中文报告见 `M5Pro_迁移M2_CPU_GPU负载影响ANE实验报告_20260805.md`。
最终折线图见 `data_cpu_corrected/results/m5pro_corrected_cpu_ane_impact_lines.pdf`。
GPU ANE Power 最终图见 `data_gpu_4096_ane_power_rerun2/results_corrected_50_replacement/m5pro_gpu4096_ane_power_corrected.pdf`。

## MPS 4096x4096 补充实验（2026-08-06）

在不改变 ANE 模型、监控口径、五档占空比、3 轮重复和完整时间线的前提下，将 GPU worker 改为 MPS FP32 4096x4096 矩阵乘。正式数据位于 `data_gpu_4096_rerun/`，15 个 trial 全部通过完整性验证。

- 100% GPU 负载：ANE power 为同轮 0% 基线的 95.16% +/- 2.88%。
- 100% GPU 负载：ANE throughput 为同轮 0% 基线的 99.01% +/- 0.57%。
- GPU 遥测在 100% 档为 162.09 +/- 16.70（监控器报告单位），证明 4096 负载有效。
- 含 round 固定效应的趋势检验：ANE power 随 GPU 负载下降（`p=0.0158`），throughput 趋势不显著（`p=0.1259`）。

复现脚本为 `scripts/run_gpu_4096_rerun.sh`，主图为 `data_gpu_4096_rerun/results/m5pro_gpu4096_ane_impact_lines.pdf`。

### ANE Power 独立重测（最终采用）

2026-08-06 使用新的随机顺序再次完成 4096x4096 GPU 五档、3 轮实验，数据位于 `data_gpu_4096_ane_power_rerun2/`。15 个 trial 全部通过完整性验证。本轮只报告 ANE Power：100% GPU 负载时为同轮 0% 基线的 96.00% +/- 1.09%，对应绝对变化 -0.945 +/- 0.370 W。含 round 固定效应的负载趋势检验为 `p=0.0173`。

50% 档第 1 轮原始点为 105.43%，而其余两轮为 100.16% 和 100.28%。按预先指定的异常点重测流程，单独重测得到 100.09%，并生成可追溯修正版 `data_gpu_4096_ane_power_rerun2/results_corrected_50_replacement/`。原始数据未删除或覆盖；替换后 50% 档为 100.18% +/- 0.24%，ANE Power 负载趋势为 `p=0.0135`。

最终 GPU ANE Power 结果以上述修正版为准。`data_gpu_2048_latest_rerun/`、`data_gpu_4096_rerun/` 和未替换的 `data_gpu_4096_ane_power_rerun2/results/` 仅作历史记录，不再作为最终 GPU ANE Power 数据。
