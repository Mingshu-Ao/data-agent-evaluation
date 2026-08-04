from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.evaluation import evaluate_run


def _load_trace_actions(task_dir: Path) -> list[str]:
    trace_path = task_dir / "trace.json"
    if not trace_path.exists():
        return []
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    return [
        str(step.get("action"))
        for step in trace.get("steps", [])
        if isinstance(step, dict) and step.get("action") is not None
    ]


def create_human_review_queue(
    *,
    run_dir: Path,
    gold_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    evaluation = evaluate_run(run_dir, gold_root)
    dataset = DABenchPublicDataset(dataset_root)
    items: list[dict[str, Any]] = []

    for task_eval in evaluation["tasks"]:
        if task_eval["exact_match"]:
            continue
        task = dataset.get_task(task_eval["task_id"])
        actions = _load_trace_actions(run_dir / task_eval["task_id"])
        items.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "question": task.question,
                "prediction_header": task_eval["prediction_header"],
                "gold_header": task_eval["gold_header"],
                "prediction_sample_rows": task_eval["prediction_sample_rows"],
                "gold_sample_rows": task_eval["gold_sample_rows"],
                "failure_reason": task_eval["failure_reason"],
                "notes": task_eval["notes"],
                "actions": actions,
                "human_questions": [
                    "Which source file/table should define the final answer columns?",
                    "Is the model using the correct filter and aggregation semantics?",
                    "Should the fix be prompt-level, operator-level, or a deterministic validation rule?",
                ],
                "suggested_label": _suggest_label(task_eval),
            }
        )

    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "dataset_root": str(dataset_root),
        "item_count": len(items),
        "items": items,
    }


def _suggest_label(task_eval: dict[str, Any]) -> str:
    notes = set(task_eval.get("notes", []))
    if "missing prediction.csv" in notes:
        return "execution_or_planning_failure"
    if "row count mismatch" in notes:
        return "filter_or_aggregation_error"
    if "header mismatch" in notes:
        return "answer_contract_error"
    if "row content mismatch" in notes:
        return "semantic_or_linking_error"
    return "needs_manual_review"


def write_human_review_queue(
    *,
    run_dir: Path,
    gold_root: Path,
    dataset_root: Path,
) -> tuple[Path, Path]:
    queue = create_human_review_queue(
        run_dir=run_dir,
        gold_root=gold_root,
        dataset_root=dataset_root,
    )
    json_path = run_dir / "human_review_queue.json"
    json_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Human Review Queue",
        "",
        f"- Run: `{run_dir}`",
        f"- Items needing review: {queue['item_count']}",
        "",
    ]
    for item in queue["items"]:
        lines.extend(
            [
                f"## {item['task_id']} - {item['suggested_label']}",
                "",
                f"- Difficulty: {item['difficulty']}",
                f"- Question: {item['question']}",
                f"- Failure reason: {item['failure_reason'] or 'None'}",
                f"- Notes: {', '.join(item['notes']) if item['notes'] else 'None'}",
                f"- Prediction header: {item['prediction_header']}",
                f"- Gold header: {item['gold_header']}",
                f"- Prediction sample rows: {item['prediction_sample_rows']}",
                f"- Gold sample rows: {item['gold_sample_rows']}",
                f"- Actions: {', '.join(item['actions'])}",
                "",
                "Questions for reviewer:",
                "",
            ]
        )
        for question in item["human_questions"]:
            lines.append(f"- [ ] {question}")
        lines.append("")
        lines.append("Reviewer note:")
        lines.append("")
        lines.append("> ")
        lines.append("")

    markdown_path = run_dir / "human_review_queue.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
