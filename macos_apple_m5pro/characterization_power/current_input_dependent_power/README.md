# 实验04：M5 Pro 黑/白像素 ANE 侧信道分类

迁移自 `manuscript/5.2.1npu.md` 的实验04，在 M5 Pro 上完成了一次正式采集。

## 实验配置

- Core ML ResNet50，输入 `B=1, [1, 3, 224, 224]`，黑色输入为全 0，白色输入为全 1。
- 推理计算单元：`CPU_AND_NE`。
- 黑色、白色各 30 次；每次为 5 s 基线、15 s 连续推理、3 s 冷却，轮次间隔 5 s。
- 监控器：M5 Pro 已有的 `ANEPowerMonitor_M5MAX_CHANNELS`；每个 CSV 保存 `Timestamp, AMCC, ANE, DCS, DRAM, FAB, GPU_Freq`。
- 每个重复只生成一个特征向量，再进行 5 折 StratifiedKFold，随机森林为 200 棵树（`random_state=42`）。交叉验证没有把同一重复的时间点拆到训练集和测试集，避免时间点级数据泄漏。

原始 ResNet50 Core ML 包在 M5 Pro 上不存在，因此依据文档中 `torch.hub.load(..., "resnet50", weights=None)` 的拓扑和随机初始化自包含重建后转换为 `ResNet50_B1_Size224_ANE.mlpackage`；它与本次采集脚本一同保存在本目录。

## M5 Pro 结果

| 指标 | 黑色 | 白色 |
|---|---:|---:|
| 活动段 ANE 监控均值 | 4.8075 | 9.0088 |
| 活动段 ANE 监控标准差（跨重复） | 0.2188 | 0.4241 |
| 活动段均值差（白−黑） | — | **4.2013** |

- 5 折准确率：`[1.0, 1.0, 1.0, 1.0, 1.0]`
- 平均准确率：`100%`，标准差：`0`
- OOF 混淆矩阵（black, white）：`[[30, 0], [0, 30]]`
- 推理吞吐约 `1,240–1,255` 次/秒。

## 文件

- `generate_resnet50_model.py`：重建并转换 B=1 ResNet50。
- `collect_experiment04.py`：M5 Pro 采集脚本。
- `analyze_experiment04.py`：特征提取、200 棵树随机森林和 5 折评估。
- `ResNet50_B1_Size224_ANE.mlpackage`：本次使用的 Core ML 模型。
- `data/`：60 个原始监控 CSV（黑/白各30个）。
- `collection_metadata.json`：采集参数和逐重复统计。
- `classification_results.json`：最终分类结果。

在 M5 Pro 上重跑：

```bash
cd ~/Desktop/m5pro_experiment04_black_white_20260806
PY=~/codex_runtime/m5pro_persistent_framework_20260801/runtime/venv/bin/python3
$PY collect_experiment04.py --reps 30 --baseline 5 --active 15 --cooldown 3 --between 5
$PY analyze_experiment04.py
```
