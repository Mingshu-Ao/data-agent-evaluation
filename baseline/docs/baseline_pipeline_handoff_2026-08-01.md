# Baseline 与 Pipeline 对接信息

日期：2026-08-01  
Baseline 负责人：敖明澍  
目标：将 KDD Cup、FDABench 等 Benchmark 转换为统一任务格式，并由同一套 Pipeline 调用不同 Data Agent，统一记录运行过程和评测结果。

## 1. 当前分工

### Baseline 侧

- 维护和运行 ReAct、DAgent-lite、AgenticData-lite、Mini-AOP。
- 接收统一格式的任务目录。
- 输出预测表格、运行轨迹、耗时和失败原因。
- 维护 KDD Phase 2 视频预处理模块。
- 对单个 Agent 的错误进行定位，例如步数上限、工具调用失败和未提交答案。

### Pipeline 侧

- 接入 KDD Cup、FDABench、KramaBench、LakeQA 等 Benchmark。
- 将不同数据集转换成统一任务目录。
- 通过统一适配器调用各个 Baseline。
- 收集预测、轨迹和资源消耗。
- 按 Benchmark 选择正确的评测指标并生成汇总报告。

### 共同负责

- 确定统一输入输出协议。
- 确定任务超时、并发数和失败分类。
- 选择小规模联调任务并验证端到端流程。
- 固定代码版本、配置和任务清单，保证实验可复现。

## 2. Baseline 当前状态

| Agent | 命令名称 | 当前状态 | 说明 |
|---|---|---|---|
| ReAct | `run-benchmark` | 已跑通 | Starter Kit 的基础 ReAct Agent |
| DAgent-lite | `run-benchmark-dagent` | 已跑通 | 论文启发的简化复现，不等同于官方完整代码 |
| AgenticData-lite | `run-benchmark-agenticdata` | 已跑通 | 包含 profile、memory、optimizer 和 validation 的简化实现 |
| Mini-AOP | `run-benchmark-mini-aop` | 已跑通 | 包含 plan-first、operator 和部分 DAG 能力的简化复现 |

当前项目目录：

```text
D:\bupt\codex_project\data_agent\kddcup\kddcup2026-data-agents-starter-kit\PHASE_1
```

Python 命令入口由 `pyproject.toml` 注册为 `dabench`。

## 3. 建议的统一输入协议 v0

Pipeline 为每个任务生成以下目录：

```text
dataset_name/
  input/
    task_1/
      task.json
      context/
        data.csv
        database.sqlite
        knowledge.md
        document.pdf
        video.mp4
  output/
    task_1/
      gold.csv
```

`task.json` 最少包含：

```json
{
  "task_id": "task_1",
  "difficulty": "unknown",
  "question": "Question text"
}
```

约束：

- `task_id` 必须与目录名一致。
- `difficulty` 可以省略，缺省值为 `unknown`。
- Agent 只能访问当前任务的 `context/`。
- `context/` 可以包含 CSV、JSON、SQLite、Markdown、PDF、图片和视频。
- 标准表格任务的金标准使用 `gold.csv`。
- 多选题、报告生成等任务可以保留额外 gold，但必须由 Pipeline 指定专用 evaluator。

## 4. 建议的统一输出协议 v0

每个任务运行后生成：

```text
run_id/
  task_1/
    prediction.csv
    trace.json
  evaluation.json
  error_analysis.md
```

`prediction.csv`：

- 第一行为列名。
- 后续行为预测结果。
- 即使只有一个值，也使用一列一行的表格形式。

`trace.json` 当前字段：

```json
{
  "task_id": "task_1",
  "answer": {
    "columns": ["field_1"],
    "rows": [["value_1"]]
  },
  "steps": [],
  "failure_reason": null,
  "succeeded": true,
  "e2e_elapsed_seconds": 52.3
}
```

每个 step 包含：

```json
{
  "step_index": 1,
  "thought": "...",
  "action": "read_csv",
  "action_input": {},
  "raw_response": "...",
  "observation": {},
  "ok": true
}
```

Pipeline 建议至少抽取以下统一字段：

| 字段 | 含义 |
|---|---|
| `benchmark` | 数据集名称 |
| `agent` | Agent 名称和版本 |
| `task_id` | 任务编号 |
| `status` | `success` 或 `failed` |
| `prediction_path` | 预测文件位置 |
| `trace_path` | 轨迹文件位置 |
| `steps` | 实际执行步数 |
| `latency_seconds` | 端到端耗时 |
| `failure_type` | 归一化失败类别 |
| `failure_reason` | 原始错误信息 |
| `model` | 使用的模型 |
| `config_id` | 配置版本 |
| `code_version` | Git commit 或版本号 |

## 5. 建议的失败分类

Pipeline 不应只统计是否生成 `prediction.csv`，建议统一为：

| 类别 | 判断依据 |
|---|---|
| `max_steps` | 达到步数上限但没有调用 `answer` |
| `timeout` | 超过单任务时间限制 |
| `model_api` | API 鉴权、限流、网络或模型返回错误 |
| `invalid_model_output` | 模型输出无法解析或不符合 action 协议 |
| `tool_error` | SQL、Python、文件读取等工具执行失败 |
| `missing_data` | 缺少上下文、gold 或必要数据库 |
| `preprocessing_error` | OCR、PDF、视频或 ASR 预处理失败 |
| `invalid_answer` | 已提交，但列数、行结构等不合法 |
| `wrong_answer` | 正常生成预测，但与 gold 不匹配 |

“运行成功”只表示成功生成预测，不等于答案正确。

## 6. Baseline 调用命令

进入项目目录并安装：

```powershell
cd D:\bupt\codex_project\data_agent\kddcup\kddcup2026-data-agents-starter-kit\PHASE_1
.\.venv-dagent\Scripts\python.exe -m pip install -e .
```

运行同一 suite：

```powershell
# ReAct
.\.venv-dagent\Scripts\dabench.exe run-benchmark `
  --config configs\kdd_phase1_compare.local.yaml `
  --suite configs\suites\phase1_coverage_20.json

# DAgent-lite
.\.venv-dagent\Scripts\dabench.exe run-benchmark-dagent `
  --config configs\kdd_phase1_compare.local.yaml `
  --suite configs\suites\phase1_coverage_20.json

# AgenticData-lite
.\.venv-dagent\Scripts\dabench.exe run-benchmark-agenticdata `
  --config configs\kdd_phase1_compare.local.yaml `
  --suite configs\suites\phase1_coverage_20.json

# Mini-AOP
.\.venv-dagent\Scripts\dabench.exe run-benchmark-mini-aop `
  --config configs\kdd_phase1_compare.local.yaml `
  --suite configs\suites\phase1_coverage_20.json
```

评测和错误诊断：

```powershell
.\.venv-dagent\Scripts\dabench.exe evaluate-run <run_dir> `
  --config <config_path>

.\.venv-dagent\Scripts\dabench.exe diagnose-step-limit <run_dir>
```

Pipeline 对接时，应将 Agent 名称映射为上述四个命令，其他参数保持统一。

## 7. 配置文件约定

配置结构：

```yaml
dataset:
  root_path: path/to/dataset/input

agent:
  model: MODEL_NAME
  api_base: API_BASE_URL
  api_key: LOCAL_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 1
  task_timeout_seconds: 600
```

安全要求：

- 只共享 `*.example.yaml`，不要共享包含真实 Key 的 `*.local.yaml`。
- API Key 不写入报告、Git 或共享文档。
- 当前代码从 YAML 读取 Key；双方应各自维护本地配置，后续可以再改为环境变量注入。
- Pipeline 的实验记录只保存模型名和 API 地址，不保存 Key。

## 8. 已有数据集与任务清单

### KDD Cup Phase 1

- 总任务：50。
- 模态：CSV、JSON、SQLite/DB、Markdown。
- 覆盖清单：`configs/suites/phase1_coverage_20.json`。
- 全量清单：`configs/suites/phase1_full_public_50.json`。

### KDD Cup Phase 2

- 总任务：60。
- 文本/结构化任务：30。
- 视频任务：30。
- 文件类型：CSV、JSON、SQLite、Markdown、PDF、MP4。
- 文本覆盖清单：`configs/suites/phase2_text_coverage_20.json`。
- 视频覆盖清单：`configs/suites/phase2_video_coverage_12.json`。
- 视频全量清单：`configs/suites/phase2_video_full_30.json`。

### FDABench-Full

- Hugging Face 总任务：2,007。
- 当前已完成 12 个 multiple-choice frozen-context replay 任务。
- 派生输入：`data/fdabench_replay_v2/input`。
- 配置：`configs/fdabench_replay_v2.local.yaml`。
- 当前缺少官方完整源数据库，因此不能声称完成官方端到端复现。

## 9. 当前可信实验结果

### KDD Phase 1 与 Phase 2 文本任务

相同 ReAct、模型、温度和步数设置，各测试 20 个任务：

| 数据集 | 任务数 | 运行成功率 | Exact match | 表头匹配率 | 行集合匹配率 | 步数上限失败 |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | 20 | 85% | 20% | 30% | 45% | 15% |
| Phase 2 | 20 | 70% | 15% | 20% | 30% | 30% |

Phase 2 的主要新增困难：

- 数据量和 gold 行数更大。
- PDF、视频和中英文混合内容增加。
- 视频条件需要与结构化字段进行跨模态对齐。
- Agent 更容易在相似表和字段之间重复探索。
- 20 任务中 6 个失败任务均属于执行查询但没有及时提交答案。

### Phase 2 视频三任务消融

| 方案 | 运行成功率 | 行集合匹配率 | 平均步数 | 平均耗时 |
|---|---:|---:|---:|---:|
| 等间隔 OCR | 100% | 0% | 22.00 | 45.31 秒 |
| 场景关键帧 OCR | 100% | 33.3% | 26.00 | 87.82 秒 |
| 场景关键帧 OCR + ASR | 100% | 33.3% | 26.33 | 84.70 秒 |

视频模块已经支持：

- PyAV 解码。
- 基于画面变化的关键帧抽取。
- 时间锚点补充，降低漏帧风险。
- RapidOCR。
- faster-whisper ASR。
- 向 VLM 发送关键帧图片的接口。

当前 DeepSeek 文本模型不接受图片，因此现有结果是关键帧 OCR 证据实验，不是完整 VLM 实验。

### FDABench 12 任务 replay

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 100% |
| Macro option F1 | 0.680 |
| Micro option F1 | 0.689 |

## 10. 可以直接提供给 Pipeline 同学的文件

建议共享：

```text
pyproject.toml
src/data_agent_baseline/
configs/*.example.yaml
configs/suites/
reports/phase_run_comparison/
reports/dataset_comparison/
reports/phase2_video/
reports/fdabench/
```

需要按服务器路径另行确认：

```text
KDD Phase 1 原始 input/output
KDD Phase 2 原始 input/output
FDABench 原始 JSONL
视频派生数据和 OCR/ASR 缓存
```

不要共享：

```text
configs/*.local.yaml
任何 API Key
包含账号密码的 SSH/VPN 配置
无版本说明的临时 artifacts
```

## 11. 双方需要确认的问题

1. Pipeline 最终采用命令行、Python 函数还是 HTTP 调用 Baseline？
2. task ID 是否由 Pipeline 重新编号，如何保留原 Benchmark ID？
3. gold 是统一 CSV，还是允许 Benchmark 专用格式？
4. 多选题、报告生成和表格问答分别采用什么 evaluator？
5. 视频和图片由 Pipeline 预处理，还是交给 Baseline 按自身策略处理？
6. token、API cost、GPU/CPU 峰值是否必须统计？
7. timeout、max_steps 和并发数由 Pipeline 统一覆盖，还是写在 Baseline 配置中？
8. 失败重试是否使用同一 run ID，如何避免重复计费？
9. 是否需要保存完整 thought/raw response；共享时是否需要脱敏？
10. 统一结果表采用 JSONL、SQLite 还是 Parquet？

## 12. 第一次联调建议

先只接 ReAct，不同时处理四个 Agent。第一轮固定使用：

`configs/suites/pipeline_smoke_phase1_easy_3.json`

| 任务 | 数据 | 能力 | 选择原因 |
|---|---|---|---|
| `task_11` | JSON + Markdown | 查询、跨文件关联、多列输出 | 已在本机单任务跑通 |
| `task_24` | CSV + JSON + Markdown | 跨文件计数 | 输出简单，容易人工核验 |
| `task_67` | CSV + JSON + Markdown | 条件筛选、平均值 | 检查数值和格式归一化 |

联调命令：

```powershell
.\.venv-dagent\Scripts\dabench.exe run-benchmark `
  --config configs\kdd_phase1_compare.local.yaml `
  --suite configs\suites\pipeline_smoke_phase1_easy_3.json
```

运行结束后保存终端输出中的 `<run_dir>`，再执行：

```powershell
.\.venv-dagent\Scripts\dabench.exe evaluate-run <run_dir> `
  --config configs\kdd_phase1_compare.local.yaml
```

第一轮工作步骤：

1. Pipeline 转换上述 3 个 KDD Phase 1 任务。
2. Baseline 读取统一目录并完成运行。
3. Pipeline 成功解析 `prediction.csv` 和 `trace.json`。
4. Pipeline 对照 gold 生成统一指标。
5. 人为制造一个超时和一个错误输出，验证失败分类。
6. 固定接口后，再依次接入 DAgent-lite、AgenticData-lite 和 Mini-AOP。
7. 最后增加 Phase 2 视频和 FDABench 专用 evaluator。

第一次联调的验收标准：

- 所有标准任务可以被 Dataset Loader 读取。
- Baseline 不需要针对 Benchmark 名称修改核心逻辑。
- 成功与失败任务都能进入统一结果表。
- 同一个 suite、模型、配置和代码版本可以重复运行。
- Pipeline 报告能够区分运行成功率、答案正确率和基础设施失败率。
