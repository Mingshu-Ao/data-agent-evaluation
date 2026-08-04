from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_trace(task_dir: Path) -> dict[str, Any]:
    trace_path = task_dir / "trace.json"
    if not trace_path.is_file():
        return {
            "task_id": task_dir.name,
            "steps": [],
            "failure_reason": "missing trace.json",
        }
    return json.loads(trace_path.read_text(encoding="utf-8"))


def _steps(trace: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [
        step
        for step in trace.get("steps", [])
        if isinstance(step, dict) and step.get("action") == action
    ]


def _observation(step: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    value = step.get("observation")
    return value if isinstance(value, dict) else {}


def analyze_agentic_data_run(run_dir: Path) -> dict[str, Any]:
    task_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("task_")
    )
    tasks: list[dict[str, Any]] = []

    for task_dir in task_dirs:
        trace = _load_trace(task_dir)
        profile_steps = _steps(trace, "__agenticdata_profile_graph__")
        plan_steps = _steps(trace, "__agenticdata_plan__")
        grammar_steps = _steps(trace, "__agenticdata_grammar_validation__")
        semantic_steps = _steps(trace, "__agenticdata_semantic_validation__")
        optimizer_steps = _steps(trace, "__agenticdata_plan_optimization__")
        memory_steps = _steps(trace, "__agenticdata_memory_retrieval__")
        fallback_steps = _steps(trace, "__agenticdata_fallback_plan__")

        profile = _observation(profile_steps[-1] if profile_steps else None).get(
            "profile_graph",
            {},
        )
        if not isinstance(profile, dict):
            profile = {}
        optimizer = _observation(
            optimizer_steps[-1] if optimizer_steps else None
        ).get("optimizer", {})
        if not isinstance(optimizer, dict):
            optimizer = {}
        memory = _observation(memory_steps[-1] if memory_steps else None)

        tasks.append(
            {
                "task_id": str(trace.get("task_id", task_dir.name)),
                "succeeded": bool(trace.get("succeeded")),
                "failure_reason": trace.get("failure_reason"),
                "profile_node_count": int(profile.get("node_count", 0) or 0),
                "profile_edge_count": int(profile.get("edge_count", 0) or 0),
                "plan_attempt_count": len(plan_steps),
                "grammar_rejection_count": sum(
                    1 for step in grammar_steps if not bool(step.get("ok"))
                ),
                "semantic_rejection_count": sum(
                    1 for step in semantic_steps if not bool(step.get("ok"))
                ),
                "fallback_used": bool(fallback_steps),
                "optimizer_used": bool(optimizer_steps),
                "estimated_cost_before": float(
                    optimizer.get("estimated_cost_before", 0.0) or 0.0
                ),
                "estimated_cost_after": float(
                    optimizer.get("estimated_cost_after", 0.0) or 0.0
                ),
                "optimizer_rules": list(optimizer.get("rules_applied", [])),
                "retrieved_memory_count": int(
                    memory.get("retrieved_count", 0) or 0
                ),
            }
        )

    optimized_tasks = [item for item in tasks if item["optimizer_used"]]
    total_before = sum(item["estimated_cost_before"] for item in optimized_tasks)
    total_after = sum(item["estimated_cost_after"] for item in optimized_tasks)
    return {
        "run_dir": str(run_dir),
        "task_count": len(tasks),
        "succeeded_task_count": sum(1 for item in tasks if item["succeeded"]),
        "tasks_with_profile_graph": sum(
            1 for item in tasks if item["profile_node_count"] > 0
        ),
        "tasks_with_plan_retries": sum(
            1 for item in tasks if item["plan_attempt_count"] > 1
        ),
        "tasks_with_fallback": sum(1 for item in tasks if item["fallback_used"]),
        "tasks_with_optimizer": len(optimized_tasks),
        "tasks_with_memory_retrieval": sum(
            1 for item in tasks if item["retrieved_memory_count"] > 0
        ),
        "estimated_cost_before": round(total_before, 3),
        "estimated_cost_after": round(total_after, 3),
        "estimated_cost_saving": round(max(total_before - total_after, 0.0), 3),
        "tasks": tasks,
    }


def write_agentic_data_report(run_dir: Path) -> tuple[Path, Path]:
    report = analyze_agentic_data_run(run_dir)
    json_path = run_dir / "agentic_data_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# AgenticData-lite Run Report",
        "",
        f"- Run: `{run_dir}`",
        f"- Tasks: {report['task_count']}",
        f"- Succeeded: {report['succeeded_task_count']}",
        f"- Tasks with profile graph: {report['tasks_with_profile_graph']}",
        f"- Tasks with plan retries: {report['tasks_with_plan_retries']}",
        f"- Tasks with fallback: {report['tasks_with_fallback']}",
        f"- Tasks with optimizer: {report['tasks_with_optimizer']}",
        f"- Tasks with memory retrieval: {report['tasks_with_memory_retrieval']}",
        f"- Proxy cost before: {report['estimated_cost_before']}",
        f"- Proxy cost after: {report['estimated_cost_after']}",
        f"- Proxy cost saving: {report['estimated_cost_saving']}",
        "",
        (
            "> Cost values are implementation-side proxies for comparison and "
            "are not the paper's measured latency or API charge."
        ),
        "",
        "## Tasks",
        "",
    ]
    for item in report["tasks"]:
        lines.extend(
            [
                f"### {item['task_id']}",
                "",
                f"- Succeeded: {item['succeeded']}",
                f"- Failure reason: {item['failure_reason'] or 'None'}",
                (
                    f"- Profile graph: {item['profile_node_count']} nodes, "
                    f"{item['profile_edge_count']} edges"
                ),
                f"- Plan attempts: {item['plan_attempt_count']}",
                f"- Grammar rejections: {item['grammar_rejection_count']}",
                f"- Semantic rejections: {item['semantic_rejection_count']}",
                f"- Fallback used: {item['fallback_used']}",
                f"- Optimizer rules: {', '.join(item['optimizer_rules']) or 'None'}",
                (
                    f"- Proxy cost: {item['estimated_cost_before']} -> "
                    f"{item['estimated_cost_after']}"
                ),
                f"- Retrieved memories: {item['retrieved_memory_count']}",
                "",
            ]
        )

    markdown_path = run_dir / "agentic_data_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
