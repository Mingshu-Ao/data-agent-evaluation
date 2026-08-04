from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.suites import profile_dataset, summarize_profiles


def _question_language(question: str) -> str:
    ascii_letters = sum(character.isascii() and character.isalpha() for character in question)
    cjk_characters = sum("\u4e00" <= character <= "\u9fff" for character in question)
    if cjk_characters and ascii_letters:
        return "mixed"
    if cjk_characters:
        return "zh"
    return "en_or_other"


def _gold_statistics(gold_root: Path) -> dict[str, Any]:
    row_counts: list[int] = []
    missing_count = 0
    for task_dir in sorted(path for path in gold_root.glob("task_*") if path.is_dir()):
        gold_path = task_dir / "gold.csv"
        if not gold_path.is_file():
            missing_count += 1
            continue
        try:
            with gold_path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
                row_count = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        except OSError:
            missing_count += 1
            continue
        row_counts.append(row_count)

    return {
        "gold_csv_count": len(row_counts),
        "missing_gold_csv_count": missing_count,
        "median_gold_rows": statistics.median(row_counts) if row_counts else 0,
        "max_gold_rows": max(row_counts, default=0),
        "tasks_with_empty_gold": sum(count == 0 for count in row_counts),
        "tasks_with_over_1000_gold_rows": sum(count > 1000 for count in row_counts),
    }


def profile_kdd_phase(input_root: Path, gold_root: Path) -> dict[str, Any]:
    dataset = DABenchPublicDataset(input_root)
    tasks = dataset.iter_tasks()
    profiles = profile_dataset(dataset)
    summary = summarize_profiles(profiles)
    summary["question_language_counts"] = dict(
        sorted(Counter(_question_language(task.question) for task in tasks).items())
    )
    summary["gold"] = _gold_statistics(gold_root)
    summary["input_root"] = str(input_root)
    summary["gold_root"] = str(gold_root)
    return summary


def compare_kdd_phases(
    *,
    phase1_input: Path,
    phase1_gold: Path,
    phase2_input: Path,
    phase2_gold: Path,
) -> dict[str, Any]:
    phase1 = profile_kdd_phase(phase1_input, phase1_gold)
    phase2 = profile_kdd_phase(phase2_input, phase2_gold)
    return {
        "phase1": phase1,
        "phase2": phase2,
        "comparison": {
            "task_count_delta": phase2["task_count"] - phase1["task_count"],
            "multimodal_task_count_delta": (
                phase2["multimodal_task_count"] - phase1["multimodal_task_count"]
            ),
            "new_phase2_file_types": sorted(
                set(phase2["file_type_counts"]) - set(phase1["file_type_counts"])
            ),
            "shared_file_types": sorted(
                set(phase2["file_type_counts"]) & set(phase1["file_type_counts"])
            ),
        },
    }


def write_kdd_phase_comparison(
    *,
    phase1_input: Path,
    phase1_gold: Path,
    phase2_input: Path,
    phase2_gold: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    report = compare_kdd_phases(
        phase1_input=phase1_input,
        phase1_gold=phase1_gold,
        phase2_input=phase2_input,
        phase2_gold=phase2_gold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kdd_phase1_phase2_comparison.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    phase1 = report["phase1"]
    phase2 = report["phase2"]
    lines = [
        "# KDD Cup Phase 1 / Phase 2 Dataset Comparison",
        "",
        "| Dimension | Phase 1 | Phase 2 |",
        "|---|---:|---:|",
        f"| Tasks | {phase1['task_count']} | {phase2['task_count']} |",
        (
            f"| Multimodal/video tasks | {phase1['multimodal_task_count']} | "
            f"{phase2['multimodal_task_count']} |"
        ),
        (
            f"| Gold CSV files | {phase1['gold']['gold_csv_count']} | "
            f"{phase2['gold']['gold_csv_count']} |"
        ),
        (
            f"| Median gold rows | {phase1['gold']['median_gold_rows']} | "
            f"{phase2['gold']['median_gold_rows']} |"
        ),
        (
            f"| Maximum gold rows | {phase1['gold']['max_gold_rows']} | "
            f"{phase2['gold']['max_gold_rows']} |"
        ),
        "",
        "## Phase 1",
        "",
        f"- Difficulties: `{phase1['difficulty_counts']}`",
        f"- File types: `{phase1['file_type_counts']}`",
        f"- Question types: `{phase1['question_type_counts']}`",
        f"- Languages: `{phase1['question_language_counts']}`",
        "",
        "## Phase 2",
        "",
        f"- Difficulties: `{phase2['difficulty_counts']}`",
        f"- File types: `{phase2['file_type_counts']}`",
        f"- Question types: `{phase2['question_type_counts']}`",
        f"- Languages: `{phase2['question_language_counts']}`",
        "",
        "## Interpretation",
        "",
        "- Phase 2 must be split into text-only and video/VLM tracks for a fair model comparison.",
        "- PDF extraction and video understanding are required capabilities in Phase 2.",
        "- Runtime success, answer correctness, and infrastructure failures should be reported separately.",
    ]
    markdown_path = output_dir / "kdd_phase1_phase2_comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
