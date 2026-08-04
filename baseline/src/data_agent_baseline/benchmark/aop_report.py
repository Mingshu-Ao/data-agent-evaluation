from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_trace(task_dir: Path) -> dict[str, Any]:
    trace_path = task_dir / "trace.json"
    if not trace_path.exists():
        return {"task_id": task_dir.name, "steps": [], "failure_reason": "missing trace.json"}
    return json.loads(trace_path.read_text(encoding="utf-8"))


def _find_step(trace: dict[str, Any], action: str) -> dict[str, Any] | None:
    for step in trace.get("steps", []):
        if isinstance(step, dict) and step.get("action") == action:
            return step
    return None


def analyze_mini_aop_run(run_dir: Path) -> dict[str, Any]:
    task_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_"))
    task_reports: list[dict[str, Any]] = []

    for task_dir in task_dirs:
        trace = _load_trace(task_dir)
        candidate_steps = [
            step
            for step in trace.get("steps", [])
            if isinstance(step, dict) and step.get("action") == "__mini_aop_candidate_plan__"
        ]
        selection_step = _find_step(trace, "__mini_aop_select_and_rewrite__")
        answer_review_step = _find_step(trace, "__mini_aop_answer_review__")

        selection_input = {}
        if selection_step is not None:
            selection_input = selection_step.get("action_input", {})
            if not isinstance(selection_input, dict):
                selection_input = {}

        metrics = selection_input.get("selection_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        dag_rewrite = selection_input.get("dag_rewrite", {})
        if not isinstance(dag_rewrite, dict):
            dag_rewrite = {}
        nodes = dag_rewrite.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []

        task_reports.append(
            {
                "task_id": task_dir.name,
                "succeeded": bool(trace.get("succeeded")),
                "failure_reason": trace.get("failure_reason"),
                "candidate_plan_count": len(candidate_steps),
                "selected_candidate_index": selection_input.get("selected_candidate_index"),
                "operator_sequence": [
                    str(node.get("op"))
                    for node in nodes
                    if isinstance(node, dict) and node.get("op") is not None
                ],
                "parallel_stages": dag_rewrite.get("parallel_stages"),
                "selection_metrics": metrics,
                "estimated_step_count": metrics.get("estimated_step_count", metrics.get("estimated_cost")),
                "estimated_operator_cost": metrics.get("estimated_operator_cost"),
                "estimated_reliability_gain": metrics.get("estimated_reliability_gain"),
                "answer_reviewed": answer_review_step is not None,
                "answer_review_changed": bool(
                    answer_review_step
                    and isinstance(answer_review_step.get("observation"), dict)
                    and isinstance(answer_review_step["observation"].get("content"), dict)
                    and answer_review_step["observation"]["content"].get("changed")
                ),
            }
        )

    reviewed_count = sum(1 for item in task_reports if item["answer_reviewed"])
    changed_count = sum(1 for item in task_reports if item["answer_review_changed"])
    return {
        "run_dir": str(run_dir),
        "task_count": len(task_reports),
        "succeeded_task_count": sum(1 for item in task_reports if item["succeeded"]),
        "tasks_with_candidate_plans": sum(1 for item in task_reports if item["candidate_plan_count"] > 0),
        "tasks_with_dag_rewrite": sum(1 for item in task_reports if item["parallel_stages"] is not None),
        "tasks_with_answer_review": reviewed_count,
        "tasks_with_answer_review_changes": changed_count,
        "tasks": task_reports,
    }


def write_mini_aop_report(run_dir: Path) -> tuple[Path, Path]:
    report = analyze_mini_aop_run(run_dir)
    json_path = run_dir / "mini_aop_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Mini-AOP Run Report",
        "",
        f"- Run: `{run_dir}`",
        f"- Tasks: {report['task_count']}",
        f"- Succeeded: {report['succeeded_task_count']}",
        f"- Tasks with candidate plans: {report['tasks_with_candidate_plans']}",
        f"- Tasks with DAG rewrite: {report['tasks_with_dag_rewrite']}",
        f"- Tasks with answer review: {report['tasks_with_answer_review']}",
        f"- Tasks with answer review changes: {report['tasks_with_answer_review_changes']}",
        "",
        "## Tasks",
        "",
    ]
    for item in report["tasks"]:
        metrics = item["selection_metrics"]
        lines.extend(
            [
                f"### {item['task_id']}",
                "",
                f"- Succeeded: {item['succeeded']}",
                f"- Failure reason: {item['failure_reason'] or 'None'}",
                f"- Candidate plans: {item['candidate_plan_count']}",
                f"- Selected candidate: {item['selected_candidate_index']}",
                f"- Operator sequence: {', '.join(item['operator_sequence']) or 'None'}",
                f"- Parallel stages: {item['parallel_stages']}",
                f"- Estimated step count: {item['estimated_step_count']}",
                f"- Estimated operator cost: {item['estimated_operator_cost']}",
                f"- Estimated reliability gain: {item['estimated_reliability_gain']}",
                f"- Answer reviewed: {item['answer_reviewed']}",
                f"- Answer review changed output: {item['answer_review_changed']}",
                "",
            ]
        )

    markdown_path = run_dir / "mini_aop_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
