# Baseline-Pipeline Integration Package

这是提供给 Pipeline 侧的脱敏 Baseline 集成包，包含 ReAct、DAgent-lite、
AgenticData-lite、Mini-AOP 的统一 CLI、测试、配置模板和 Phase 1 三任务 smoke suite。

## 对接决定 v0

1. 第一版通过 `dabench` 命令行调用 Baseline。
2. 保留原 Benchmark 的 `task_id`，使用 `(benchmark, task_id)` 作为全局标识。
3. 表格问答默认使用 `gold.csv`；多选题和报告任务允许专用 gold/evaluator。
4. Pipeline 生成统一任务目录并保存原始媒体；Baseline 负责关键帧、OCR、ASR、VLM 等实验相关预处理。
5. Pipeline 生成运行 YAML 并设置并发和超时；Baseline 按配置执行。
6. 重试保留原失败轨迹，不覆盖已有 run。
7. 完整数据、视频、模型缓存和运行 artifacts 不进入 GitHub。

## 文件

```text
pyproject.toml
src/data_agent_baseline/
tests/
configs/baseline_pipeline.example.yaml
configs/suites/pipeline_smoke_phase1_easy_3.json
docs/baseline_pipeline_handoff_2026-08-01.md
```

## 安装与测试

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

视频和 ASR 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,video-asr]"
```

## 运行 smoke suite

复制并填写本地配置，真实 API Key 只保存在 local 文件中：

```powershell
Copy-Item configs\baseline_pipeline.example.yaml configs\baseline_pipeline.local.yaml

.\.venv\Scripts\dabench.exe run-benchmark `
  --config configs\baseline_pipeline.local.yaml `
  --suite configs\suites\pipeline_smoke_phase1_easy_3.json
```

Agent 命令映射：

```text
react       -> dabench run-benchmark
dagent      -> dabench run-benchmark-dagent
agenticdata -> dabench run-benchmark-agenticdata
mini-aop    -> dabench run-benchmark-mini-aop
```

当前 DAgent-lite、AgenticData-lite、Mini-AOP 属于论文启发的简化复现，不等同于官方完整实现。
