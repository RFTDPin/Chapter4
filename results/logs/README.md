# 运行日志

该目录用于存放在线实验与补偿集构建的日志，包括：

- `*_latency.csv`: 每帧延迟统计（ms）
- `*_tracker.jsonl`: 帧级状态机记录（符合 `fvessel/schemas/experiment_log_schema.json`）
- `compensation_set.json`: Day1 阶段生成的补偿索引
