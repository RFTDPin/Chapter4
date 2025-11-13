# FVessel 数据与先验目录

该目录用于放置 FVessel 多模态数据及校准结果，按如下约定组织：

```
fvessel/
├── raw/               # 原始视频帧、AIS CSV、YOLO 标注（按 clip 划分）
├── processed/         # 经过切片/抽帧后的缓存（按 split 划分）
├── priors/            # 逐帧先验 JSON（光学- AIS 对齐结果）
├── calibration/       # 单应与时间偏置估计脚本
└── schemas/           # 结构化 JSON schema（便于复现）
```

## 数据放置

- **视频帧**：`raw/<scene_id>/frames/*.jpg`
- **AIS 报文**：`raw/<scene_id>/ais.csv`（至少包含 `timestamp`, `mmsi`, `lon`, `lat`, `sog`, `cog`）
- **YOLO 标注**：`raw/<scene_id>/labels/*.txt`
- **帧索引**：`raw/<scene_id>/frame_index.csv`（列：`frame_id`, `timestamp`）

## 生成的文件

- `calib/<scene_id>_homography.json`: 单应矩阵 H 与元信息
- `calib/<scene_id>_time_offset.json`: 视频/ AIS 时间偏置
- `priors/<scene_id>.jsonl`: 逐帧投影先验（(u,v,σ_r,speed,course,confidence)）

所有 JSON/JSONL 均需满足 `schemas/` 中的定义。运行脚本示例：

```bash
python -m multimodal_tracker.scripts.prepare_data \
    --scene-id harbor_01 \
    --dataset-root fvessel/raw \
    --output-root fvessel \
    --fps 25 \
    --img-width 1920 --img-height 1080
```

运行前请确认本地已安装 `ultralytics`, `opencv-python`, `pyproj`（可选）等依赖。
