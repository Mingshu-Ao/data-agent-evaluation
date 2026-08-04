from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task_id: str
    succeeded: bool
    prediction_exists: bool
    gold_exists: bool
    exact_match: bool
    header_match: bool
    row_set_match: bool
    prediction_rows: int
    gold_rows: int
    prediction_header: list[str]
    gold_header: list[str]
    prediction_sample_rows: list[list[str]]
    gold_sample_rows: list[list[str]]
    actions: list[str]
    failure_reason: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "succeeded": self.succeeded,
            "prediction_exists": self.prediction_exists,
            "gold_exists": self.gold_exists,
            "exact_match": self.exact_match,
            "header_match": self.header_match,
            "row_set_match": self.row_set_match,
            "prediction_rows": self.prediction_rows,
            "gold_rows": self.gold_rows,
            "prediction_header": self.prediction_header,
            "gold_header": self.gold_header,
            "prediction_sample_rows": self.prediction_sample_rows,
            "gold_sample_rows": self.gold_sample_rows,
            "actions": self.actions,
            "failure_reason": self.failure_reason,
            "notes": self.notes,
        }


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def _normalize_rows(rows: list[list[str]]) -> list[tuple[str, ...]]:
    return [tuple(cell.strip() for cell in row) for row in rows]


def _evaluate_task(task_dir: Path, gold_root: Path) -> TaskEvaluation:
    task_id = task_dir.name
    prediction_path = task_dir / "prediction.csv"
    trace_path = task_dir / "trace.json"
    gold_path = gold_root / task_id / "gold.csv"
    notes: list[str] = []

    trace: dict[str, Any] = {}
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    else:
        notes.append("missing trace.json")

    actions = [str(step.get("action")) for step in trace.get("steps", []) if isinstance(step, dict)]
    failure_reason = trace.get("failure_reason")
    succeeded = bool(trace.get("succeeded"))

    prediction_exists = prediction_path.exists()
    gold_exists = gold_path.exists()
    if not prediction_exists:
        notes.append("missing prediction.csv")
    if not gold_exists:
        notes.append("missing gold.csv")

    pred_rows: list[list[str]] = []
    gold_rows: list[list[str]] = []
    if prediction_exists:
        pred_rows = _read_csv_rows(prediction_path)
    if gold_exists:
        gold_rows = _read_csv_rows(gold_path)

    pred_norm = _normalize_rows(pred_rows)
    gold_norm = _normalize_rows(gold_rows)
    header_match = bool(pred_norm and gold_norm and pred_norm[0] == gold_norm[0])
    row_set_match = bool(pred_norm and gold_norm and set(pred_norm[1:]) == set(gold_norm[1:]))
    exact_match = pred_norm == gold_norm

    if prediction_exists and gold_exists:
        if not header_match:
            notes.append("header mismatch")
        if len(pred_norm) - 1 != len(gold_norm) - 1:
            notes.append("row count mismatch")
        if header_match and not row_set_match:
            notes.append("row content mismatch")
        if row_set_match and not exact_match:
            notes.append("row order or formatting mismatch")

    return TaskEvaluation(
        task_id=task_id,
        succeeded=succeeded,
        prediction_exists=prediction_exists,
        gold_exists=gold_exists,
        exact_match=exact_match,
        header_match=header_match,
        row_set_match=row_set_match,
        prediction_rows=max(len(pred_rows) - 1, 0),
        gold_rows=max(len(gold_rows) - 1, 0),
        prediction_header=pred_rows[0] if pred_rows else [],
        gold_header=gold_rows[0] if gold_rows else [],
        prediction_sample_rows=pred_rows[1:4],
        gold_sample_rows=gold_rows[1:4],
        actions=actions,
        failure_reason=str(failure_reason) if failure_reason else None,
        notes=notes,
    )


def evaluate_run(run_dir: Path, gold_root: Path) -> dict[str, Any]:
    task_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_"))
    task_results = [_evaluate_task(task_dir, gold_root) for task_dir in task_dirs]

    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "task_count": len(task_results),
        "succeeded_task_count": sum(1 for item in task_results if item.succeeded),
        "exact_match_count": sum(1 for item in task_results if item.exact_match),
        "header_match_count": sum(1 for item in task_results if item.header_match),
        "row_set_match_count": sum(1 for item in task_results if item.row_set_match),
        "tasks": [item.to_dict() for item in task_results],
    }


def write_evaluation_outputs(run_dir: Path, gold_root: Path) -> tuple[Path, Path]:
    result = evaluate_run(run_dir, gold_root)
    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Error Analysis",
        "",
        f"- Run: `{run_dir}`",
        f"- Gold root: `{gold_root}`",
        f"- Tasks: {result['task_count']}",
        f"- Succeeded: {result['succeeded_task_count']}",
        f"- Exact match: {result['exact_match_count']}",
        f"- Header match: {result['header_match_count']}",
        f"- Row-set match: {result['row_set_match_count']}",
        "",
        "## Tasks",
        "",
    ]
    for item in result["tasks"]:
        status = "exact" if item["exact_match"] else "needs_analysis"
        report_lines.extend(
            [
                f"### {item['task_id']} - {status}",
                "",
                f"- Succeeded: {item['succeeded']}",
                f"- Prediction rows: {item['prediction_rows']}",
                f"- Gold rows: {item['gold_rows']}",
                f"- Prediction header: {item['prediction_header']}",
                f"- Gold header: {item['gold_header']}",
                f"- Prediction sample rows: {item['prediction_sample_rows']}",
                f"- Gold sample rows: {item['gold_sample_rows']}",
                f"- Header match: {item['header_match']}",
                f"- Row-set match: {item['row_set_match']}",
                f"- Actions: {', '.join(item['actions'])}",
                f"- Failure reason: {item['failure_reason'] or 'None'}",
                f"- Notes: {', '.join(item['notes']) if item['notes'] else 'None'}",
                "",
            ]
        )

    report_path = run_dir / "error_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return evaluation_path, report_path


def compare_runs(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    gold_root: Path,
) -> dict[str, Any]:
    baseline = evaluate_run(baseline_run_dir, gold_root)
    candidate = evaluate_run(candidate_run_dir, gold_root)
    baseline_tasks = {item["task_id"]: item for item in baseline["tasks"]}
    candidate_tasks = {item["task_id"]: item for item in candidate["tasks"]}
    task_ids = sorted(set(baseline_tasks) | set(candidate_tasks))

    comparisons: list[dict[str, Any]] = []
    improved = regressed = unchanged_correct = unchanged_wrong = 0
    for task_id in task_ids:
        base_item = baseline_tasks.get(task_id)
        cand_item = candidate_tasks.get(task_id)
        base_exact = bool(base_item and base_item["exact_match"])
        cand_exact = bool(cand_item and cand_item["exact_match"])
        if not base_exact and cand_exact:
            status = "improved"
            improved += 1
        elif base_exact and not cand_exact:
            status = "regressed"
            regressed += 1
        elif base_exact and cand_exact:
            status = "unchanged_correct"
            unchanged_correct += 1
        else:
            status = "unchanged_wrong"
            unchanged_wrong += 1
        comparisons.append(
            {
                "task_id": task_id,
                "status": status,
                "baseline_exact": base_exact,
                "candidate_exact": cand_exact,
                "baseline_notes": base_item["notes"] if base_item else ["missing baseline task"],
                "candidate_notes": cand_item["notes"] if cand_item else ["missing candidate task"],
            }
        )

    return {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "gold_root": str(gold_root),
        "task_count": len(task_ids),
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_correct_count": unchanged_correct,
        "unchanged_wrong_count": unchanged_wrong,
        "baseline_exact_match_count": baseline["exact_match_count"],
        "candidate_exact_match_count": candidate["exact_match_count"],
        "tasks": comparisons,
    }
