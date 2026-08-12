# data-agent-pipeline

Data Agent 统一评测 Pipeline（KDD Cup 2026 Data Agent 赛道）。

本目录是本评测管线的源码，与同仓库的 `baseline/`（4 个 Data Agent 的 Baseline 实现）平级。
管线的职责是：把不同 Benchmark（KDD Cup / FDAbench / KramaBench / LakeQA）的数据统一成
Baseline 可读的格式，调用 Baseline Agent 评测，收集输出并生成统一评分报告。

## 目录结构

```
data-agent-pipeline/
├── eval_config.py          # API 配置（Key 只从环境变量读取，仓库内无真实 Key）
├── eval_pipeline.py        # 主入口（--benchmark kdd|fdabench|krama|lakeqa）
├── suite_loader.py         # 读取 suite JSON 任务清单
├── baseline_adapter.py     # 调用 Baseline Agent（subprocess + 临时 config 注入）+ TaskIdMapper
├── run_kdd_scale.py        # 多 agent KDD 规模跑驱动（独立 work_dir + 对比表）
├── test_*.py               # 单测 / 联通测试 / 冒烟脚本
├── pipeline_smoke_*.json   # 各 benchmark 冒烟 suite
├── eval_knowledge.json     # 评测知识库（mock 模式注入）
│
├── loaders/                # 数据加载层（统一 list_tasks / load_task / task_count）
│   ├── kdd_loader.py       #   KDD Cup Phase 1+2
│   ├── fdabench_loader.py  #   FDAbench-Full (668 任务)
│   ├── krama_loader.py     #   KramaBench (104 任务)
│   └── lakeqa_loader.py    #   LakeQA (1141 任务)
│
├── workers/                # 评测 Worker（CrewAI Flow 架构）
│   ├── structured_query.py #   结构化查询（CSV/JSON/SQLite/Markdown）
│   ├── python_analysis.py  #   pandas 数据分析
│   ├── answer_worker.py    #   答案生成（防幻觉 + 知识库注入）
│   ├── score_worker.py     #   KDD 评分（Recall/Final，行相等）
│   └── judge_worker.py     #   LLM 语义评分（Krama/LakeQA/FDAbench 自由文本）
│
├── vector_store/           # 非结构化数据处理（PDF/Word/图片/视频 → 语义检索）
│   └── retrieval.py        #   ingest_and_retrieve 编排
│
└── scripts/                # 数据下载脚本
    ├── download_kramabench.py
    └── download_lakeqa.py
```

## 依赖与环境

- Python 3.12，`crewai`、`pydantic`、`datasets`、`huggingface_hub`、`pyyaml`；
  可选 `av`(pyav)、`pymupdf`、`python-docx`（视频/PDF/Word 处理）。
- Baseline 环境：同仓库 `../baseline/`（独立 `.venv`，`pip install -e ".[dev]"`）。
- 本目录的 `data/`（数据集 symlink/缓存）与 `pipeline_runs/`（运行产物）不入库，
  需自行准备或运行 `scripts/download_*.py` 获取。

### API Key 配置

仓库内不保存任何真实 Key。运行前设置环境变量（任设一个即可同时作为 Commander/Worker 的 Key）：

```bash
export DEEPSEEK_COMMANDER_API_KEY=sk-xxx
export DEEPSEEK_WORKER_API_KEY=sk-xxx
# 可选，默认 https://api.deepseek.com
export DEEPSEEK_API_BASE_URL=https://api.deepseek.com
```

未设置时对应 LLM 调用会明确报错（预期行为）。

## 用法

```bash
# Mock 模式（KDD，不需要 Baseline，但需 TAS 可导入 + API key）
python eval_pipeline.py --suite pipeline_smoke_phase1_easy_3.json --benchmark kdd --mock

# 真实 Baseline（注意：--baseline-project 用正斜杠路径）
python eval_pipeline.py --suite pipeline_smoke_phase1_easy_3.json --agent react \
  --baseline-project C:/path/to/data-agent-evaluation/baseline

# 其他 benchmark（数据就绪后）
python eval_pipeline.py --suite pipeline_smoke_krama_2.json  --benchmark krama  --mock
python eval_pipeline.py --suite pipeline_smoke_lakeqa_2.json --benchmark lakeqa --mock

# FDAbench（原始 FDA ID，自动走 ID 映射层映射为 task_<int> 喂 Baseline）
python eval_pipeline.py --suite pipeline_smoke_fdabench_2.json --benchmark fdabench \
  --agent react --baseline-project C:/path/to/data-agent-evaluation/baseline

# 可用 agent：--agent react / dagent-lite / agenticdata-lite / mini-aop

# 单测 / 冒烟
python test_id_mapping.py      # ID 映射层（无网络）
python test_judge_smoke.py     # LLM 语义评分（真实 API）
```

## 与 Baseline 的对接协议

**统一输入：** `dataset/input/task_N/task.json + context/` 和 `dataset/output/task_N/gold.csv`
（Baseline 校验 task.json 必须恰好 `{task_id, difficulty, question}` 三键；目录名必须是 `task_<int>`）

**统一输出（evaluation.json）：** 每条结果 13 字段
`benchmark, agent, task_id, status, prediction_path, trace_path, steps, latency_seconds,
failure_type, failure_reason, model, config_id, code_version` + `summary`
（total / run_success / correct / wrong_answer / infra_failed / avg_final_score / scores）

**失败分类：** 9 类 + 2 个 Pipeline 哨兵（`missing_trace`, `unknown`）。

**ID 映射：** 非 `task_<int>` 的 benchmark ID（`FDA0001` / `legal-hard-1` / `lakeqa-full:EQA...`）
由 `TaskIdMapper` 映射为 `task_{100000+i}` 再喂给 Baseline；对外 records 仍用原始 ID。

**评分：** KDD 表格走 `ScoreFlow` 行相等（0.99 阈值判 correct）；
Krama / LakeQA / FDAbench 的自由文本/数值答案走 `JudgeFlow` LLM 语义评分（`final_score∈[0,1]`）。
