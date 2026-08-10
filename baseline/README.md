# Baseline-Pipeline Integration Package

这是 2026-08-10 整理的真实联调包，供 Pipeline 统一调用四种 Data Agent：

- ReAct
- DAgent-lite
- AgenticData-lite
- Mini-AOP

仓库包含最新 Baseline 代码、A1 输出契约检查、无密钥配置、Phase 1 三任务
smoke suite，以及成功/失败输出样例。真实 Benchmark 输入、gold、模型密钥和
运行产物不在公开仓库中提供。

DAgent-lite、AgenticData-lite 和 Mini-AOP 是论文启发的简化复现，不等同于论文
官方完整实现。

## 目录

```text
src/data_agent_baseline/                 Baseline 实现
tests/                                   离线单元测试
configs/baseline_pipeline.example.yaml   无密钥配置模板
configs/suites/                          固定任务清单
examples/                                Pipeline 解析样例
docs/integration_contract.md             输入输出和统计协议
docs/baseline_pipeline_handoff.md        详细背景说明
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

API Key 通过环境变量提供，不写入配置：

```powershell
$env:DEEPSEEK_API_KEY = "你的API Key"
Copy-Item configs\baseline_pipeline.example.yaml configs\baseline_pipeline.local.yaml
```

## 三任务真实联调

运行前，请在本地准备以下目录；这些数据不要提交到公开仓库：

```text
data/phase1_smoke/input/task_11
data/phase1_smoke/input/task_24
data/phase1_smoke/input/task_67
data/phase1_smoke/output/task_11/gold.csv
data/phase1_smoke/output/task_24/gold.csv
data/phase1_smoke/output/task_67/gold.csv
```

先运行 ReAct：

```powershell
.\.venv\Scripts\dabench.exe run-benchmark `
  --config configs\baseline_pipeline.local.yaml `
  --suite configs\suites\pipeline_smoke_phase1_easy_3.json
```

其他 Agent 仅替换命令：

| Agent | 命令 |
|---|---|
| ReAct | `run-benchmark` |
| DAgent-lite | `run-benchmark-dagent` |
| AgenticData-lite | `run-benchmark-agenticdata` |
| Mini-AOP | `run-benchmark-mini-aop` |

## A0 与 A1

- A0：`answer_contract.enabled: false`，原始 Baseline。
- A1：`answer_contract.enabled: true`，增加统一输出契约检查和一次模型复核。

A1 是实验变量，不是第五种 Agent。Pipeline 建议增加 `variant` 字段，记录 `A0`
或 `A1`。启用 A1 后，每个生成答案的任务还会输出 `answer_contract.json`。

## 验收重点

1. 在本地放置 Benchmark 数据后，Pipeline 能读取三个真实任务并调用四种 Agent。
2. Pipeline 能解析 `summary.json`、`prediction.csv` 和 `trace.json`。
3. 运行成功率与官方答案准确率分别统计。
4. 所有实验记录模型、步数上限、超时、配置版本和代码版本。
5. API Key、服务器账号和本地配置不得进入 GitHub 或运行报告。
