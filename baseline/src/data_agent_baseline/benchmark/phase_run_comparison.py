from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.evaluation import evaluate_run


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _failure_category(reason: str | None) -> str:
    text = (reason or "").lower()
    if not text:
        return "missing_or_unknown"
    if "max_steps" in text or "step limit" in text or "maximum steps" in text:
        return "max_steps"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if any(token in text for token in ("connection", "api", "http", "token", "model")):
        return "model_or_api"
    if any(token in text for token in ("tool", "parse", "json", "csv", "validation")):
        return "tool_or_format"
    return "other"


def _summarize_phase(
    *,
    label: str,
    run_dir: Path,
    gold_root: Path,
) -> dict[str, Any]:
    evaluation = evaluate_run(run_dir, gold_root)
    total = int(evaluation["task_count"])
    succeeded = int(evaluation["succeeded_task_count"])
    exact = int(evaluation["exact_match_count"])
    header = int(evaluation["header_match_count"])
    row_set = int(evaluation["row_set_match_count"])
    failure_counts = Counter(
        _failure_category(item.get("failure_reason"))
        for item in evaluation["tasks"]
        if not item["succeeded"]
    )
    return {
        "label": label,
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "task_count": total,
        "succeeded_task_count": succeeded,
        "run_success_rate": _rate(succeeded, total),
        "exact_match_count": exact,
        "exact_match_rate": _rate(exact, total),
        "exact_among_succeeded_rate": _rate(exact, succeeded),
        "header_match_count": header,
        "header_match_rate": _rate(header, total),
        "row_set_match_count": row_set,
        "row_set_match_rate": _rate(row_set, total),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failed_tasks": [
            {
                "task_id": item["task_id"],
                "failure_category": _failure_category(item.get("failure_reason")),
                "failure_reason": item.get("failure_reason"),
                "notes": item.get("notes", []),
            }
            for item in evaluation["tasks"]
            if not item["succeeded"]
        ],
    }


def compare_phase_runs(
    *,
    phase1_run_dir: Path,
    phase1_gold_root: Path,
    phase2_run_dir: Path,
    phase2_gold_root: Path,
    phase1_label: str = "KDD Phase 1",
    phase2_label: str = "KDD Phase 2",
) -> dict[str, Any]:
    phase1 = _summarize_phase(
        label=phase1_label,
        run_dir=phase1_run_dir,
        gold_root=phase1_gold_root,
    )
    phase2 = _summarize_phase(
        label=phase2_label,
        run_dir=phase2_run_dir,
        gold_root=phase2_gold_root,
    )
    return {
        "comparison_note": (
            "Run success measures whether a task produced a prediction. Exact/header/row-set "
            "rates measure correctness against each phase's own gold outputs."
        ),
        "phase1": phase1,
        "phase2": phase2,
        "rate_delta_phase2_minus_phase1": {
            "run_success_rate": round(
                phase2["run_success_rate"] - phase1["run_success_rate"], 4
            ),
            "exact_match_rate": round(
                phase2["exact_match_rate"] - phase1["exact_match_rate"], 4
            ),
            "header_match_rate": round(
                phase2["header_match_rate"] - phase1["header_match_rate"], 4
            ),
            "row_set_match_rate": round(
                phase2["row_set_match_rate"] - phase1["row_set_match_rate"], 4
            ),
        },
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_phase_run_comparison(
    *,
    phase1_run_dir: Path,
    phase1_gold_root: Path,
    phase2_run_dir: Path,
    phase2_gold_root: Path,
    output_dir: Path,
    phase1_label: str = "KDD Phase 1",
    phase2_label: str = "KDD Phase 2",
) -> tuple[Path, Path]:
    result = compare_phase_runs(
        phase1_run_dir=phase1_run_dir,
        phase1_gold_root=phase1_gold_root,
        phase2_run_dir=phase2_run_dir,
        phase2_gold_root=phase2_gold_root,
        phase1_label=phase1_label,
        phase2_label=phase2_label,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kdd_phase_run_comparison.json"
    markdown_path = output_dir / "kdd_phase_run_comparison.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    phase1 = result["phase1"]
    phase2 = result["phase2"]
    delta = result["rate_delta_phase2_minus_phase1"]
    lines = [
        "# KDD Phase 1 / Phase 2 Run Comparison",
        "",
        (
            "Success rate means a prediction file was produced. Accuracy metrics are "
            "computed against each phase's own gold outputs."
        ),
        "",
        "| Dataset | Tasks | Run success | Exact match | Header match | Row-set match |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {phase1['label']} | {phase1['task_count']} | "
            f"{phase1['succeeded_task_count']} ({_percent(phase1['run_success_rate'])}) | "
            f"{phase1['exact_match_count']} ({_percent(phase1['exact_match_rate'])}) | "
            f"{phase1['header_match_count']} ({_percent(phase1['header_match_rate'])}) | "
            f"{phase1['row_set_match_count']} ({_percent(phase1['row_set_match_rate'])}) |"
        ),
        (
            f"| {phase2['label']} | {phase2['task_count']} | "
            f"{phase2['succeeded_task_count']} ({_percent(phase2['run_success_rate'])}) | "
            f"{phase2['exact_match_count']} ({_percent(phase2['exact_match_rate'])}) | "
            f"{phase2['header_match_count']} ({_percent(phase2['header_match_rate'])}) | "
            f"{phase2['row_set_match_count']} ({_percent(phase2['row_set_match_rate'])}) |"
        ),
        "",
        "## Phase 2 Minus Phase 1",
        "",
        f"- Run success rate: {_percent(delta['run_success_rate'])}",
        f"- Exact-match rate: {_percent(delta['exact_match_rate'])}",
        f"- Header-match rate: {_percent(delta['header_match_rate'])}",
        f"- Row-set-match rate: {_percent(delta['row_set_match_rate'])}",
        "",
        "## Failures",
        "",
        f"- {phase1['label']}: {phase1['failure_counts'] or 'None'}",
        f"- {phase2['label']}: {phase2['failure_counts'] or 'None'}",
        "",
        "## Interpretation",
        "",
        "- Compare runs made with the same agent, model, temperature, and step limit.",
        "- Report Phase 2 video tasks separately when the text model cannot consume video.",
        "- A successful run is not necessarily correct; use exact and row-set rates together.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
