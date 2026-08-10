# Baseline-Pipeline Contract v1

## Ownership

| Item | Owner |
|---|---|
| Benchmark loading and unified task layout | Pipeline |
| ReAct, DAgent-lite, AgenticData-lite, Mini-AOP | Baseline |
| Benchmark evaluator and aggregate report | Pipeline |
| Agent trace and internal failure diagnosis | Baseline |
| Timeout, concurrency and repeat count | Pipeline |
| Agent max steps and agent-specific options | Baseline configuration |
| Phase 2 video preprocessing implementation | Baseline |
| Selection and reuse of a fixed preprocessed dataset | Pipeline |

## Input

```text
dataset/
  input/task_N/task.json
  input/task_N/context/*
  output/task_N/gold.csv
```

Keep the original task ID. Use `(benchmark, task_id)` as the global identifier.

## Output

```text
run_id/
  summary.json
  task_N/
    prediction.csv
    trace.json
    answer_contract.json  # A1 only
```

Pipeline should collect:

| Field | Meaning |
|---|---|
| `benchmark` | Dataset name and version |
| `agent` | Agent name |
| `variant` | `A0` or `A1` |
| `task_id` | Original task ID |
| `attempt_id` | Retry identifier |
| `prediction_generated` | Whether prediction.csv exists |
| `official_passed` | Whether the benchmark evaluator passes |
| `steps` | Executed steps |
| `latency_seconds` | End-to-end latency |
| `failure_type` | Normalized failure category |
| `failure_reason` | Original error |
| `model` | Model name |
| `max_steps` | Step budget |
| `timeout_seconds` | Task timeout |
| `config_id` | Configuration version |
| `code_version` | Git commit or release ID |

`prediction_generated=true` does not imply `official_passed=true`.

## Failure types

Use: `max_steps`, `timeout`, `model_api`, `invalid_model_output`, `tool_error`,
`missing_data`, `preprocessing_error`, `invalid_answer`, and `wrong_answer`.

## Retry

Do not overwrite the first attempt. Keep a new `attempt_id` and preserve both traces.
The initial report should show first-attempt accuracy; repeated runs can additionally show
mean, variation and pass@k.

## Video fairness

Generate one fixed derived dataset for each preprocessing method, such as scene-keyframe
OCR. All agents in the same comparison must read the same derived evidence. Record the
preprocessing method and version in the result table.

