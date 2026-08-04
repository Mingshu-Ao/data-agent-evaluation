from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("report", "single", "multiple")
COMMON_REQUIRED_FIELDS = {
    "database_type",
    "db",
    "instance_id",
    "level",
    "query",
    "question_type",
    "task_id",
    "tools_available",
}


@dataclass(frozen=True, slots=True)
class FDABenchRecord:
    split: str
    payload: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.payload["task_id"])

    @property
    def level(self) -> str:
        return str(self.payload.get("level") or "unknown")

    @property
    def database_type(self) -> str:
        return str(self.payload.get("database_type") or "unknown")

    @property
    def question_type(self) -> str:
        return str(self.payload.get("question_type") or self.split)

    @property
    def source_signature(self) -> str:
        tools = {
            str(tool)
            for tool in self.payload.get("tools_available", [])
            if isinstance(tool, str)
        }
        sources = []
        if "execute_sql" in tools:
            sources.append("sql")
        if "web_search" in tools:
            sources.append("web")
        if "vector_search" in tools:
            sources.append("vector")
        return "+".join(sources) or "other"


def _read_jsonl(path: Path, split: str) -> list[FDABenchRecord]:
    records: list[FDABenchRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object.")
            missing = COMMON_REQUIRED_FIELDS - set(payload)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            records.append(FDABenchRecord(split=split, payload=payload))
    return records


def load_fdabench(root: Path, *, splits: tuple[str, ...] = SPLITS) -> list[FDABenchRecord]:
    records: list[FDABenchRecord] = []
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown FDABench split: {split}")
        path = root / split / "data.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FDABench split: {path}")
        records.extend(_read_jsonl(path, split))
    return records


def summarize_fdabench(records: list[FDABenchRecord]) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    for record in records:
        tool_counts.update(
            str(tool)
            for tool in record.payload.get("tools_available", [])
            if isinstance(tool, str)
        )
    return {
        "task_count": len(records),
        "split_counts": dict(sorted(Counter(record.split for record in records).items())),
        "level_counts": dict(sorted(Counter(record.level for record in records).items())),
        "database_type_counts": dict(
            sorted(Counter(record.database_type for record in records).items())
        ),
        "question_type_counts": dict(
            sorted(Counter(record.question_type for record in records).items())
        ),
        "source_signature_counts": dict(
            sorted(Counter(record.source_signature for record in records).items())
        ),
        "tool_counts": dict(sorted(tool_counts.items())),
        "released_answer_count": sum(
            record.payload.get("correct_answer") not in (None, "")
            for record in records
        ),
        "report_with_ground_truth_count": sum(
            record.split == "report"
            and bool(record.payload.get("ground_truth_report"))
            for record in records
        ),
        "report_with_sql_result_count": sum(
            record.split == "report" and record.payload.get("sql_result") not in (None, "")
            for record in records
        ),
        "tasks_with_frozen_web_count": sum(
            bool(record.payload.get("frozen_web_search")) for record in records
        ),
        "tasks_with_frozen_vector_count": sum(
            bool(record.payload.get("frozen_vector_search")) for record in records
        ),
    }


def select_fdabench_coverage(
    records: list[FDABenchRecord],
    *,
    size: int,
) -> list[FDABenchRecord]:
    if size <= 0:
        raise ValueError("size must be positive.")
    if size >= len(records):
        return list(records)

    selected: list[FDABenchRecord] = []
    selected_keys: set[tuple[str, str]] = set()

    def add(record: FDABenchRecord) -> None:
        key = (record.split, record.task_id)
        if key not in selected_keys and len(selected) < size:
            selected.append(record)
            selected_keys.add(key)

    dimensions = (
        lambda record: record.split,
        lambda record: record.level,
        lambda record: record.database_type,
        lambda record: record.source_signature,
    )
    for dimension in dimensions:
        groups: dict[str, list[FDABenchRecord]] = {}
        for record in records:
            groups.setdefault(dimension(record), []).append(record)
        for _, candidates in sorted(groups.items()):
            add(candidates[0])
            if len(selected) >= size:
                break
        if len(selected) >= size:
            break

    for record in records:
        add(record)
        if len(selected) >= size:
            break
    return selected


def write_fdabench_report_and_suite(
    *,
    root: Path,
    output_dir: Path,
    suite_size: int,
) -> tuple[Path, Path, Path]:
    records = load_fdabench(root)
    summary = summarize_fdabench(records)
    selected = select_fdabench_coverage(records, size=suite_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "fdabench_dataset_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    suite_payload = {
        "suite_name": f"fdabench_coverage_{suite_size}",
        "description": (
            "Coverage suite spanning FDABench splits, levels, database types, "
            "and required source combinations."
        ),
        "dataset_summary": summary,
        "suite_summary": summarize_fdabench(selected),
        "tasks": [
            {
                "split": record.split,
                "task_id": record.task_id,
                "instance_id": record.payload.get("instance_id"),
                "level": record.level,
                "database_type": record.database_type,
                "question_type": record.question_type,
                "source_signature": record.source_signature,
                "query": record.payload.get("query"),
                "has_released_answer": record.payload.get("correct_answer")
                not in (None, ""),
            }
            for record in selected
        ],
    }
    suite_path = output_dir / f"fdabench_coverage_{suite_size}.json"
    suite_path.write_text(
        json.dumps(suite_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# FDABench Dataset Readiness Report",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Splits: `{summary['split_counts']}`",
        f"- Levels: `{summary['level_counts']}`",
        f"- Database types: `{summary['database_type_counts']}`",
        f"- Source combinations: `{summary['source_signature_counts']}`",
        f"- Publicly released choice answers: {summary['released_answer_count']}",
        f"- Report tasks with reference reports: {summary['report_with_ground_truth_count']}",
        f"- Report tasks with SQL results: {summary['report_with_sql_result_count']}",
        "",
        "## Evaluation Boundary",
        "",
        "- Report tasks require SQL, web retrieval, vector retrieval, and rubric-based report scoring.",
        "- The public metadata does not bundle all 139 source databases.",
        "- Single-choice gold answers are not released in the public split.",
        "- Full scoring therefore requires the official resources or submission workflow.",
    ]
    markdown_path = output_dir / "fdabench_dataset_report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return report_path, markdown_path, suite_path


def materialize_fdabench_multiple_replay(
    *,
    root: Path,
    output_root: Path,
    size: int,
) -> tuple[Path, Path, Path]:
    records = [
        record
        for record in load_fdabench(root, splits=("multiple",))
        if record.payload.get("correct_answer") not in (None, "")
    ]
    selected = select_fdabench_coverage(records, size=size)
    input_root = output_root / "input"
    gold_root = output_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    gold_root.mkdir(parents=True, exist_ok=True)
    manifest_tasks: list[dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        local_task_id = f"task_{index}"
        context_dir = input_root / local_task_id / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        options = record.payload.get("options", {})
        option_text = "\n".join(
            f"{label}: {text}"
            for label, text in options.items()
        )
        question = (
            f"{record.payload.get('query')}\n\nOptions:\n{option_text}\n\n"
            "This is a multiple-select question: one or more options may be correct. "
            "Evaluate every option independently using the released frozen evidence, "
            "and return all supported correct option labels. Return the labels in a "
            "single column named answer, with one label per row and no explanation."
        )
        (context_dir.parent / "task.json").write_text(
            json.dumps(
                {
                    "task_id": local_task_id,
                    "difficulty": record.level,
                    "question": question,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for field in (
            "frozen_web_search",
            "frozen_vector_search",
            "gold_subtasks",
            "options",
        ):
            (context_dir / f"{field}.json").write_text(
                json.dumps(
                    record.payload.get(field),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (context_dir / "README.md").write_text(
            "# FDABench frozen-context replay\n\n"
            "This replay task exposes the benchmark's released frozen retrieval and "
            "gold-subtask artifacts. It does not include the original source database, "
            "so it evaluates evidence integration and option reasoning rather than "
            "end-to-end SQL execution.\n",
            encoding="utf-8",
        )

        gold_task_dir = gold_root / local_task_id
        gold_task_dir.mkdir(parents=True, exist_ok=True)
        answers = record.payload.get("correct_answer", [])
        if not isinstance(answers, list):
            answers = [answers]
        with (gold_task_dir / "gold.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["answer"])
            writer.writerows([[answer] for answer in answers])

        manifest_tasks.append(
            {
                "local_task_id": local_task_id,
                "fdabench_task_id": record.task_id,
                "instance_id": record.payload.get("instance_id"),
                "database": record.payload.get("db"),
                "level": record.level,
                "database_type": record.database_type,
                "source_signature": record.source_signature,
                "correct_answer_count": len(answers),
            }
        )

    manifest_path = output_root / "replay_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "frozen_context_replay",
                "limitations": [
                    "Original source databases are not bundled.",
                    "Gold subtasks expose expected SQL/results when publicly released.",
                    "Scores measure evidence integration, not end-to-end database analysis.",
                ],
                "task_count": len(manifest_tasks),
                "tasks": manifest_tasks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return input_root, gold_root, manifest_path


def _read_answer_labels(path: Path) -> tuple[set[str], bool]:
    if not path.is_file():
        return set(), False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return set(), False
        answer_field = "answer" if "answer" in fieldnames else fieldnames[0]
        labels = {
            str(row.get(answer_field, "")).strip().upper()
            for row in reader
            if str(row.get(answer_field, "")).strip()
        }
    return labels, fieldnames == ["answer"]


def evaluate_fdabench_replay(
    *,
    run_dir: Path,
    gold_root: Path,
) -> dict[str, Any]:
    task_ids = sorted(
        (path.name for path in gold_root.iterdir() if path.is_dir()),
        key=lambda task_id: int(task_id.rsplit("_", 1)[-1]),
    )
    task_results: list[dict[str, Any]] = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    for task_id in task_ids:
        prediction_path = run_dir / task_id / "prediction.csv"
        gold_path = gold_root / task_id / "gold.csv"
        prediction, header_match = _read_answer_labels(prediction_path)
        gold, _ = _read_answer_labels(gold_path)
        true_positive = len(prediction & gold)
        false_positive = len(prediction - gold)
        false_negative = len(gold - prediction)
        precision = (
            true_positive / (true_positive + false_positive)
            if prediction
            else 0.0
        )
        recall = true_positive / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative
        task_results.append(
            {
                "task_id": task_id,
                "succeeded": prediction_path.is_file(),
                "header_match": header_match,
                "exact_match": prediction == gold,
                "prediction": sorted(prediction),
                "gold": sorted(gold),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "option_precision": round(precision, 6),
                "option_recall": round(recall, 6),
                "option_f1": round(f1, 6),
            }
        )

    task_count = len(task_results)
    micro_precision = (
        total_true_positive / (total_true_positive + total_false_positive)
        if total_true_positive + total_false_positive
        else 0.0
    )
    micro_recall = (
        total_true_positive / (total_true_positive + total_false_negative)
        if total_true_positive + total_false_negative
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "task_count": task_count,
        "succeeded_task_count": sum(item["succeeded"] for item in task_results),
        "header_match_count": sum(item["header_match"] for item in task_results),
        "exact_match_count": sum(item["exact_match"] for item in task_results),
        "macro_option_precision": round(
            sum(item["option_precision"] for item in task_results) / task_count,
            6,
        )
        if task_count
        else 0.0,
        "macro_option_recall": round(
            sum(item["option_recall"] for item in task_results) / task_count,
            6,
        )
        if task_count
        else 0.0,
        "macro_option_f1": round(
            sum(item["option_f1"] for item in task_results) / task_count,
            6,
        )
        if task_count
        else 0.0,
        "micro_option_precision": round(micro_precision, 6),
        "micro_option_recall": round(micro_recall, 6),
        "micro_option_f1": round(micro_f1, 6),
        "tasks": task_results,
    }


def write_fdabench_replay_evaluation(
    *,
    run_dir: Path,
    gold_root: Path,
) -> tuple[Path, Path]:
    evaluation = evaluate_fdabench_replay(run_dir=run_dir, gold_root=gold_root)
    json_path = run_dir / "fdabench_replay_evaluation.json"
    json_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = run_dir / "fdabench_replay_evaluation.md"
    lines = [
        "# FDABench Replay Evaluation",
        "",
        f"- Tasks: {evaluation['task_count']}",
        f"- Succeeded: {evaluation['succeeded_task_count']}",
        f"- Exact matches: {evaluation['exact_match_count']}",
        f"- Header matches: {evaluation['header_match_count']}",
        f"- Macro option precision: {evaluation['macro_option_precision']:.3f}",
        f"- Macro option recall: {evaluation['macro_option_recall']:.3f}",
        f"- Macro option F1: {evaluation['macro_option_f1']:.3f}",
        f"- Micro option precision: {evaluation['micro_option_precision']:.3f}",
        f"- Micro option recall: {evaluation['micro_option_recall']:.3f}",
        f"- Micro option F1: {evaluation['micro_option_f1']:.3f}",
        "",
        "| Task | Prediction | Gold | Precision | Recall | F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in evaluation["tasks"]:
        lines.append(
            f"| {item['task_id']} | {','.join(item['prediction'])} | "
            f"{','.join(item['gold'])} | {item['option_precision']:.3f} | "
            f"{item['option_recall']:.3f} | {item['option_f1']:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
