# 跨模态船舶跟踪实验计划（Day 1 摘要）

本文件追踪 7 天交付计划的 Day 1 进展，覆盖数据准备、校准与基线评测。

## 今日目标

1. 完成 FVessel 数据目录搭建与放置约定。
2. 实现单应估计、时间偏置估计与 AIS→像素投影脚本。
3. 输出逐帧先验 (`priors/*.jsonl`)、校准文件 (`calib/*.json`) 与 YOLOv11 基线入口。
4. 给出补偿集索引生成脚本，为 Day3–Day4 补偿方案评估提供输入。

## 结果文件

| 文件 | 说明 |
| --- | --- |
| `fvessel/calibration/estimate_homography.py` | 基于匹配点的 DLT 单应估计实现 |
| `fvessel/calibration/estimate_time_offset.py` | AIS-视频时间偏置估计（速度序列相关） |
| `multimodal_tracker/scripts/prepare_data.py` | 统一入口：调用校准并生成逐帧先验 |
| `multimodal_tracker/scripts/run_baseline_detection.py` | YOLOv11-s Baseline 指标导出 |
| `multimodal_tracker/scripts/generate_compensation_set.py` | 基于 priors + detections 的补偿集筛选 |
| `fvessel/schemas/*.json` | JSON 结构约束，保证复现一致性 |

## 后续依赖

- Day 2 的增量匹配器将直接消费 `priors/<scene_id>.jsonl` 与 `results/tables/baseline_detection.csv`。
- Day 3/4 需要 `results/logs/` 中的补偿集输出。

请在运行任何脚本前设置好 `PYTHONPATH=/workspace/Chapter4` 并确认本地安装 `torch>=2.1`, `ultralytics>=8.1`, `opencv-python`, `pandas`, `numpy`, `tqdm`。
