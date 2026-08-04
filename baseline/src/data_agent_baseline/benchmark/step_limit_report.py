from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

EXECUTION_ACTIONS = {
    "execute_context_sql",
    "execute_python",
    "execute_structured_sql",
}
EXPLORATION_ACTIONS = {
    "inspect_sqlite_schema",
    "list_context",
    "profile_context",
    "read_csv",
    "read_doc",
    "read_json",
    "search_documents",
}


def _signature(step: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": step.get("action"),
            "action_input": step.get("action_input", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _max_consecutive_repeat(signatures: list[str]) -> int:
    longest = 0
    current = 0
    previous: str | None = None
    for signature in signatures:
        if signature == previous:
            current += 1
        else:
            previous = signature
            current = 1
        longest = max(longest, current)
    return longest


def diagnose_step_limit_trace(trace: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in trace.get("steps", []) if isinstance(step, dict)]
    runtime_steps = [
        step
        for step in steps
        if not str(step.get("action", "")).startswith("__")
    ]
    actions = [str(step.get("action", "")) for step in runtime_steps]
    signatures = [_signature(step) for step in runtime_steps]
    signature_counts = Counter(signatures)
    failed_steps = sum(not bool(step.get("ok")) for step in runtime_steps)
    error_steps = sum(action == "__error__" for action in actions)
    exploration_steps = sum(action in EXPLORATION_ACTIONS for action in actions)
    execution_steps = sum(action in EXECUTION_ACTIONS for action in actions)
    answer_attempts = sum(action == "answer" for action in actions)
    max_repeat = max(signature_counts.values(), default=0)
    consecutive_repeat = _max_consecutive_repeat(signatures)

    if error_steps >= 2 or (runtime_steps and failed_steps / len(runtime_steps) >= 0.4):
        category = "tool_or_parse_error_loop"
    elif answer_attempts:
        category = "answer_rejected_or_invalid"
    elif consecutive_repeat >= 3 or max_repeat >= 4:
        category = "repeated_action_loop"
    elif execution_steps == 0 and exploration_steps:
        category = "exploration_without_execution"
    elif execution_steps:
        category = "execution_without_submission"
    else:
        category = "planning_or_unknown_stall"

    return {
        "task_id": trace.get("task_id"),
        "failure_reason": trace.get("failure_reason"),
        "category": category,
        "total_steps": len(steps),
        "runtime_steps": len(runtime_steps),
        "failed_steps": failed_steps,
        "exploration_steps": exploration_steps,
        "execution_steps": execution_steps,
        "answer_attempts": answer_attempts,
        "unique_action_signatures": len(signature_counts),
        "max_identical_action_count": max_repeat,
        "max_consecutive_repeat": consecutive_repeat,
        "action_counts": dict(sorted(Counter(actions).items())),
        "last_actions": actions[-5:],
    }


def analyze_step_limit_run(run_dir: Path) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in run_dir.glob("task_*") if path.is_dir()):
        trace_path = task_dir / "trace.json"
        if not trace_path.is_file():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        failure_reason = str(trace.get("failure_reason") or "")
        if "max_steps" not in failure_reason:
            continue
        tasks.append(diagnose_step_limit_trace(trace))

    categories = Counter(item["category"] for item in tasks)
    return {
        "run_dir": str(run_dir),
        "step_limit_task_count": len(tasks),
        "category_counts": dict(sorted(categories.items())),
        "tasks": tasks,
    }


def write_step_limit_report(run_dir: Path) -> tuple[Path, Path]:
    report = analyze_step_limit_run(run_dir)
    json_path = run_dir / "step_limit_diagnosis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Step-limit Diagnosis",
        "",
        f"- Run: `{run_dir}`",
        f"- Step-limit tasks: {report['step_limit_task_count']}",
        f"- Categories: `{report['category_counts']}`",
        "",
        "## Task Details",
        "",
    ]
    for item in report["tasks"]:
        lines.extend(
            [
                f"### {item['task_id']}",
                "",
                f"- Category: `{item['category']}`",
                f"- Runtime steps: {item['runtime_steps']}",
                f"- Failed steps: {item['failed_steps']}",
                f"- Exploration / execution: {item['exploration_steps']} / {item['execution_steps']}",
                f"- Repeated identical action: {item['max_identical_action_count']}",
                f"- Last actions: `{item['last_actions']}`",
                "",
            ]
        )
    markdown_path = run_dir / "step_limit_diagnosis.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
