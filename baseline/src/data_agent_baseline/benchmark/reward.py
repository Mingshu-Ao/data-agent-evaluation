from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.aop_report import analyze_mini_aop_run
from data_agent_baseline.benchmark.evaluation import evaluate_run


def _task_reward(task_eval: dict[str, Any], trace_task: dict[str, Any] | None) -> dict[str, Any]:
    reward = 0.0
    components: dict[str, float] = {}

    if task_eval.get("succeeded"):
        components["submitted_answer"] = 0.10
    if task_eval.get("header_match"):
        components["header_match"] = 0.20
    if task_eval.get("row_set_match"):
        components["row_set_match"] = 0.45
    if task_eval.get("exact_match"):
        components["exact_match"] = 1.00

    reward += sum(components.values())

    prediction_rows = int(task_eval.get("prediction_rows") or 0)
    gold_rows = int(task_eval.get("gold_rows") or 0)
    row_count_delta = abs(prediction_rows - gold_rows)
    if row_count_delta:
        penalty = min(0.20, row_count_delta * 0.03)
        components["row_count_penalty"] = -penalty
        reward -= penalty

    if not task_eval.get("prediction_exists"):
        components["missing_prediction_penalty"] = -0.40
        reward -= 0.40

    if trace_task is not None:
        if trace_task.get("answer_reviewed"):
            components["answer_review_bonus"] = 0.05
            reward += 0.05
        if trace_task.get("parallel_stages") is not None:
            components["dag_rewrite_bonus"] = 0.05
            reward += 0.05
        candidate_count = int(trace_task.get("candidate_plan_count") or 0)
        if candidate_count > 0:
            components["candidate_plan_bonus"] = min(0.06, candidate_count * 0.02)
            reward += components["candidate_plan_bonus"]

    reward = max(0.0, min(1.0, reward))
    return {
        "task_id": task_eval["task_id"],
        "reward": round(reward, 4),
        "components": components,
        "exact_match": bool(task_eval.get("exact_match")),
        "notes": task_eval.get("notes", []),
    }


def score_run_rewards(run_dir: Path, gold_root: Path) -> dict[str, Any]:
    evaluation = evaluate_run(run_dir, gold_root)
    trace_report = analyze_mini_aop_run(run_dir)
    trace_by_task = {item["task_id"]: item for item in trace_report["tasks"]}
    task_rewards = [
        _task_reward(item, trace_by_task.get(item["task_id"]))
        for item in evaluation["tasks"]
    ]
    average_reward = sum(item["reward"] for item in task_rewards) / len(task_rewards) if task_rewards else 0.0
    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "task_count": len(task_rewards),
        "average_reward": round(average_reward, 4),
        "exact_match_count": evaluation["exact_match_count"],
        "row_set_match_count": evaluation["row_set_match_count"],
        "tasks": task_rewards,
    }


def write_reward_report(run_dir: Path, gold_root: Path) -> tuple[Path, Path]:
    reward = score_run_rewards(run_dir, gold_root)
    json_path = run_dir / "reward_report.json"
    json_path.write_text(json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Reward Report",
        "",
        f"- Run: `{run_dir}`",
        f"- Tasks: {reward['task_count']}",
        f"- Average reward: {reward['average_reward']}",
        f"- Exact match: {reward['exact_match_count']}",
        f"- Row-set match: {reward['row_set_match_count']}",
        "",
        "## Tasks",
        "",
    ]
    for item in reward["tasks"]:
        lines.extend(
            [
                f"### {item['task_id']}",
                "",
                f"- Reward: {item['reward']}",
                f"- Exact match: {item['exact_match']}",
                f"- Notes: {', '.join(item['notes']) if item['notes'] else 'None'}",
                f"- Components: {item['components']}",
                "",
            ]
        )

    markdown_path = run_dir / "reward_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
