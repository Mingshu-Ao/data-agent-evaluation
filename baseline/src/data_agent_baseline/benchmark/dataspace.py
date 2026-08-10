from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.benchmark.suites import write_suite

TASK_ID_PATTERN = re.compile(r"^task_(\d+)$")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

STRUCTURED_EXTENSIONS = {
    "csv",
    "db",
    "json",
    "parquet",
    "sqlite",
    "sqlite3",
    "xls",
    "xlsx",
}
SQLITE_EXTENSIONS = {"db", "sqlite", "sqlite3"}
PDF_EXTENSIONS = {"pdf"}
VIDEO_EXTENSIONS = {"avi", "m4v", "mkv", "mov", "mp4", "webm"}
IMAGE_EXTENSIONS = {"bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"}
AUDIO_EXTENSIONS = {"aac", "flac", "m4a", "mp3", "ogg", "wav"}
TEXT_DOCUMENT_EXTENSIONS = {"doc", "docx", "md", "rtf", "txt"}

GROUP_ORDER = ("structured", "join", "pdf", "video", "mixed_modal")


def _task_number(task_id: str) -> int:
    match = TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Invalid DataSpace task id: {task_id}")
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class DataSpaceLayout:
    benchmark_root: Path
    input_root: Path
    gold_root: Path
    config_root: Path


def resolve_dataspace_layout(benchmark_root: Path) -> DataSpaceLayout:
    root = benchmark_root.resolve()
    candidates = [root, root / "DataSpace-Benchmark", root / "release" / "DataSpace-Benchmark"]
    for candidate in candidates:
        if (candidate / "input").is_dir():
            return DataSpaceLayout(
                benchmark_root=candidate,
                input_root=candidate / "input",
                gold_root=candidate / "output",
                config_root=candidate / "evaluation" / "configs",
            )
    raise FileNotFoundError(
        "Cannot find DataSpace-Benchmark/input under "
        f"{root}. Pass the extracted benchmark directory or its parent."
    )


def _detect_language(text: str) -> str:
    has_cjk = CJK_PATTERN.search(text) is not None
    has_latin = LATIN_PATTERN.search(text) is not None
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def _sqlite_table_count(path: Path) -> int:
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()
    except (OSError, sqlite3.Error):
        return 0
    return int(row[0]) if row else 0


def _visible_context_files(task: PublicTask) -> list[Path]:
    return [
        path
        for path in sorted(task.context_dir.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and path.name.lower() not in {"desktop.ini", "thumbs.db"}
    ]


@dataclass(frozen=True, slots=True)
class DataSpaceTaskProfile:
    task_id: str
    question: str
    language: str
    file_count: int
    file_types: tuple[str, ...]
    structured_source_count: int
    sqlite_table_count: int
    has_pdf: bool
    has_video: bool
    has_image: bool
    has_audio: bool
    has_text_document: bool
    structured: bool
    join: bool
    mixed_modal: bool
    public_reference: bool

    @property
    def groups(self) -> tuple[str, ...]:
        values = {
            "structured": self.structured,
            "join": self.join,
            "pdf": self.has_pdf,
            "video": self.has_video,
            "mixed_modal": self.mixed_modal,
        }
        return tuple(group for group in GROUP_ORDER if values[group])

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "language": self.language,
            "file_count": self.file_count,
            "file_types": list(self.file_types),
            "structured_source_count": self.structured_source_count,
            "sqlite_table_count": self.sqlite_table_count,
            "has_pdf": self.has_pdf,
            "has_video": self.has_video,
            "has_image": self.has_image,
            "has_audio": self.has_audio,
            "has_text_document": self.has_text_document,
            "structured": self.structured,
            "join": self.join,
            "mixed_modal": self.mixed_modal,
            "groups": list(self.groups),
            "public_reference": self.public_reference,
        }


def profile_dataspace_task(
    task: PublicTask,
    *,
    gold_root: Path,
    config_root: Path,
) -> DataSpaceTaskProfile:
    files = _visible_context_files(task)
    suffixes = [path.suffix.lower().lstrip(".") or "other" for path in files]
    file_types = tuple(sorted(set(suffixes)))
    structured_files = [
        path for path, suffix in zip(files, suffixes, strict=True) if suffix in STRUCTURED_EXTENSIONS
    ]
    sqlite_tables = sum(
        _sqlite_table_count(path)
        for path, suffix in zip(files, suffixes, strict=True)
        if suffix in SQLITE_EXTENSIONS
    )
    has_pdf = bool(set(file_types) & PDF_EXTENSIONS)
    has_video = bool(set(file_types) & VIDEO_EXTENSIONS)
    has_image = bool(set(file_types) & IMAGE_EXTENSIONS)
    has_audio = bool(set(file_types) & AUDIO_EXTENSIONS)
    has_text_document = bool(set(file_types) & TEXT_DOCUMENT_EXTENSIONS)

    modality_families = {
        family
        for family, present in (
            ("structured", bool(structured_files)),
            ("pdf", has_pdf),
            ("video", has_video),
            ("image", has_image),
            ("audio", has_audio),
            ("text_document", has_text_document),
        )
        if present
    }
    public_reference = (
        (gold_root / task.task_id / "gold.csv").is_file()
        and (config_root / f"{task.task_id}.json").is_file()
    )
    return DataSpaceTaskProfile(
        task_id=task.task_id,
        question=task.question,
        language=_detect_language(task.question),
        file_count=len(files),
        file_types=file_types,
        structured_source_count=len(structured_files),
        sqlite_table_count=sqlite_tables,
        has_pdf=has_pdf,
        has_video=has_video,
        has_image=has_image,
        has_audio=has_audio,
        has_text_document=has_text_document,
        structured=bool(structured_files) and not (has_pdf or has_video or has_image or has_audio),
        join=len(structured_files) >= 2 or sqlite_tables >= 2,
        mixed_modal=len(modality_families) >= 2,
        public_reference=public_reference,
    )


def profile_dataspace(benchmark_root: Path) -> tuple[DataSpaceLayout, list[DataSpaceTaskProfile]]:
    layout = resolve_dataspace_layout(benchmark_root)
    dataset = DABenchPublicDataset(layout.input_root)
    profiles = [
        profile_dataspace_task(
            task,
            gold_root=layout.gold_root,
            config_root=layout.config_root,
        )
        for task in dataset.iter_tasks()
    ]
    return layout, profiles


def summarize_dataspace_profiles(profiles: list[DataSpaceTaskProfile]) -> dict[str, Any]:
    file_type_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for profile in profiles:
        file_type_counts.update(profile.file_types)
        group_counts.update(profile.groups)
    return {
        "task_count": len(profiles),
        "public_reference_count": sum(profile.public_reference for profile in profiles),
        "language_counts": dict(sorted(Counter(profile.language for profile in profiles).items())),
        "file_type_counts": dict(sorted(file_type_counts.items())),
        "group_counts": {group: group_counts[group] for group in GROUP_ORDER},
        "total_context_file_count": sum(profile.file_count for profile in profiles),
    }


def _coverage_features(profile: DataSpaceTaskProfile) -> set[str]:
    return {
        *(f"group:{group}" for group in profile.groups),
        *(f"file:{file_type}" for file_type in profile.file_types),
        f"language:{profile.language}",
    }


def select_dataspace_coverage_suite(
    profiles: list[DataSpaceTaskProfile],
    *,
    suite_size: int,
) -> list[DataSpaceTaskProfile]:
    if suite_size <= 0:
        raise ValueError("suite_size must be positive.")
    ordered = sorted(profiles, key=lambda profile: _task_number(profile.task_id))
    if suite_size >= len(ordered):
        return ordered

    selected: list[DataSpaceTaskProfile] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()

    def add(profile: DataSpaceTaskProfile) -> None:
        selected.append(profile)
        selected_ids.add(profile.task_id)
        covered.update(_coverage_features(profile))

    for group in GROUP_ORDER:
        candidate = next(
            (
                profile
                for profile in ordered
                if profile.task_id not in selected_ids and group in profile.groups
            ),
            None,
        )
        if candidate is not None:
            add(candidate)
        if len(selected) >= suite_size:
            break

    while len(selected) < suite_size:
        remaining = [profile for profile in ordered if profile.task_id not in selected_ids]
        candidate = max(
            remaining,
            key=lambda profile: (
                len(_coverage_features(profile) - covered),
                -_task_number(profile.task_id),
            ),
        )
        add(candidate)

    return sorted(selected, key=lambda profile: _task_number(profile.task_id))


def build_dataspace_suite_payload(
    *,
    profiles: list[DataSpaceTaskProfile],
    suite_name: str,
    suite_size: int,
    description: str,
) -> dict[str, Any]:
    public_profiles = [profile for profile in profiles if profile.public_reference]
    if not public_profiles:
        raise ValueError("No DataSpace public-reference tasks were found.")
    selected = select_dataspace_coverage_suite(public_profiles, suite_size=suite_size)
    return {
        "suite_name": suite_name,
        "description": description,
        "selection_policy": (
            "Deterministic greedy coverage over the non-exclusive structured, join, PDF, "
            "video, and mixed-modal groups, then file types and question language."
        ),
        "group_definition": {
            "structured": "Has structured data and no PDF, video, image, or audio.",
            "join": "Has at least two structured files or at least two SQLite user tables.",
            "pdf": "Contains at least one PDF file.",
            "video": "Contains at least one video file.",
            "mixed_modal": "Contains at least two evidence families.",
            "note": "Groups are heuristic and non-exclusive; official scores are unchanged.",
        },
        "public_summary": summarize_dataspace_profiles(public_profiles),
        "suite_summary": summarize_dataspace_profiles(selected),
        "task_ids": [profile.task_id for profile in selected],
        "tasks": [profile.to_dict() for profile in selected],
    }


def _render_dataset_report(
    *,
    layout: DataSpaceLayout,
    summary: dict[str, Any],
    public_profiles: list[DataSpaceTaskProfile],
) -> str:
    lines = [
        "# DataSpace Dataset Report",
        "",
        f"- Benchmark root: `{layout.benchmark_root}`",
        f"- Total tasks discovered: {summary['task_count']}",
        f"- Public-reference tasks: {summary['public_reference_count']}",
        f"- Context files discovered: {summary['total_context_file_count']}",
        "- Group labels are heuristic, non-exclusive analysis slices.",
        "",
        "## Group coverage",
        "",
        "| Group | Public tasks |",
        "|---|---:|",
    ]
    public_group_counts = summarize_dataspace_profiles(public_profiles)["group_counts"]
    lines.extend(f"| {group} | {public_group_counts[group]} |" for group in GROUP_ORDER)
    lines.extend(
        [
            "",
            "## Public task inventory",
            "",
            "| Task | Language | Files | Groups | File types |",
            "|---|---|---:|---|---|",
        ]
    )
    for profile in public_profiles:
        lines.append(
            f"| {profile.task_id} | {profile.language} | {profile.file_count} | "
            f"{', '.join(profile.groups) or '-'} | {', '.join(profile.file_types) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_dataspace_report_and_suites(
    *,
    benchmark_root: Path,
    report_dir: Path,
    suites_dir: Path,
    smoke_size: int = 5,
    coverage_size: int = 20,
) -> dict[str, Path]:
    layout, profiles = profile_dataspace(benchmark_root)
    public_profiles = [profile for profile in profiles if profile.public_reference]
    public_profiles.sort(key=lambda profile: _task_number(profile.task_id))
    summary = summarize_dataspace_profiles(profiles)

    report_dir.mkdir(parents=True, exist_ok=True)
    suites_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "dataspace_dataset_report.json"
    report_md = report_dir / "dataspace_dataset_report.md"
    report_json.write_text(
        json.dumps(
            {
                "benchmark_root": str(layout.benchmark_root),
                "summary": summary,
                "tasks": [profile.to_dict() for profile in profiles],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_md.write_text(
        _render_dataset_report(
            layout=layout,
            summary=summary,
            public_profiles=public_profiles,
        ),
        encoding="utf-8",
    )

    suite_specs = (
        (
            "smoke",
            smoke_size,
            "Small DataSpace public-reference smoke test covering the main evidence groups.",
        ),
        (
            "coverage",
            coverage_size,
            "DataSpace public-reference coverage suite for preflight comparison.",
        ),
        (
            "public",
            len(public_profiles),
            "All locally scorable DataSpace public-reference tasks.",
        ),
    )
    paths: dict[str, Path] = {"report_json": report_json, "report_md": report_md}
    for label, size, description in suite_specs:
        payload = build_dataspace_suite_payload(
            profiles=profiles,
            suite_name=f"dataspace_{label}_{size}",
            suite_size=size,
            description=description,
        )
        path = suites_dir / f"dataspace_{label}_{size}.json"
        write_suite(path, payload)
        paths[f"suite_{label}"] = path

    for group in GROUP_ORDER:
        group_profiles = [profile for profile in public_profiles if group in profile.groups]
        if not group_profiles:
            continue
        payload = build_dataspace_suite_payload(
            profiles=group_profiles,
            suite_name=f"dataspace_{group}_{len(group_profiles)}",
            suite_size=len(group_profiles),
            description=f"All public-reference DataSpace tasks in the {group} analysis slice.",
        )
        path = suites_dir / f"dataspace_{group}_{len(group_profiles)}.json"
        write_suite(path, payload)
        paths[f"suite_group_{group}"] = path

    if len(public_profiles) >= 2:
        adapt_size = min(coverage_size, len(public_profiles) - 1)
        adapt_profiles = select_dataspace_coverage_suite(
            public_profiles,
            suite_size=adapt_size,
        )
        adapt_ids = {profile.task_id for profile in adapt_profiles}
        test_profiles = [
            profile for profile in public_profiles if profile.task_id not in adapt_ids
        ]
        split_specs = (
            (
                "ace_adapt",
                adapt_profiles,
                "ACE-lite adaptation tasks. Their official outcomes may update the playbook.",
            ),
            (
                "ace_test",
                test_profiles,
                "Held-out ACE-lite test tasks. Never curate the playbook from these outcomes.",
            ),
        )
        for label, split_profiles, description in split_specs:
            payload = build_dataspace_suite_payload(
                profiles=split_profiles,
                suite_name=f"dataspace_{label}_{len(split_profiles)}",
                suite_size=len(split_profiles),
                description=description,
            )
            payload["split_protocol"] = {
                "adapt_task_count": len(adapt_profiles),
                "held_out_test_task_count": len(test_profiles),
                "disjoint": True,
            }
            path = suites_dir / f"dataspace_{label}_{len(split_profiles)}.json"
            write_suite(path, payload)
            paths[f"suite_{label}"] = path
    return paths


def _evaluation_config_root(
    *,
    run_dir: Path,
    source_config_root: Path,
    task_ids: list[str] | None,
) -> Path:
    if task_ids is None:
        return source_config_root
    normalized_ids = sorted(set(task_ids), key=_task_number)
    digest = hashlib.sha256("\n".join(normalized_ids).encode("utf-8")).hexdigest()[:12]
    subset_root = run_dir / "dataspace_evaluation_inputs" / digest
    subset_root.mkdir(parents=True, exist_ok=True)
    for task_id in normalized_ids:
        source = source_config_root / f"{task_id}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing public evaluation config: {source}")
        shutil.copy2(source, subset_root / source.name)
    return subset_root


def run_official_dataspace_evaluator(
    *,
    run_dir: Path,
    benchmark_root: Path,
    evaluator_script: Path,
    task_ids: list[str] | None = None,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    evaluator_script = evaluator_script.resolve()
    if not evaluator_script.is_file():
        raise FileNotFoundError(f"DataSpace evaluator does not exist: {evaluator_script}")
    layout = resolve_dataspace_layout(benchmark_root)
    config_root = _evaluation_config_root(
        run_dir=run_dir,
        source_config_root=layout.config_root,
        task_ids=task_ids,
    )
    output = (output_path or run_dir / "dataspace_evaluation.json").resolve()
    completed = subprocess.run(
        (
            sys.executable,
            str(evaluator_script),
            "--prediction-root",
            str(run_dir),
            "--gold-root",
            str(layout.gold_root),
            "--config-root",
            str(config_root),
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Official DataSpace evaluator failed with exit code {completed.returncode}: {detail}"
        )
    return json.loads(output.read_text(encoding="utf-8")), output


def build_dataspace_group_analysis(
    *,
    evaluation: dict[str, Any],
    profiles: list[DataSpaceTaskProfile],
) -> dict[str, Any]:
    profiles_by_id = {profile.task_id: profile for profile in profiles}
    task_results = {
        str(task["task_id"]): task
        for task in evaluation.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }

    groups: dict[str, Any] = {}
    for group in GROUP_ORDER:
        ids = sorted(
            (
                task_id
                for task_id in task_results
                if task_id in profiles_by_id and group in profiles_by_id[task_id].groups
            ),
            key=_task_number,
        )
        passed = sum(bool(task_results[task_id].get("passed")) for task_id in ids)
        errors = Counter(
            str(task_results[task_id].get("error_code") or "wrong_answer")
            for task_id in ids
            if not task_results[task_id].get("passed")
        )
        groups[group] = {
            "task_count": len(ids),
            "passed_task_count": passed,
            "task_accuracy": passed / len(ids) if ids else None,
            "error_counts": dict(sorted(errors.items())),
            "task_ids": ids,
        }

    languages: dict[str, Any] = {}
    for language in sorted({profile.language for profile in profiles_by_id.values()}):
        ids = sorted(
            (
                task_id
                for task_id in task_results
                if task_id in profiles_by_id and profiles_by_id[task_id].language == language
            ),
            key=_task_number,
        )
        if not ids:
            continue
        passed = sum(bool(task_results[task_id].get("passed")) for task_id in ids)
        languages[language] = {
            "task_count": len(ids),
            "passed_task_count": passed,
            "task_accuracy": passed / len(ids),
        }

    return {
        "metric": evaluation.get("metric"),
        "overall": {
            "task_count": evaluation.get("task_count", 0),
            "passed_task_count": evaluation.get("passed_task_count", 0),
            "task_accuracy": evaluation.get("task_accuracy", 0.0),
        },
        "groups_are_non_exclusive": True,
        "groups": groups,
        "languages": languages,
    }


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_dataspace_group_analysis(analysis: dict[str, Any]) -> str:
    overall = analysis["overall"]
    lines = [
        "# DataSpace Group Analysis",
        "",
        f"- Official metric: `{analysis['metric']}`",
        (
            f"- Overall: {overall['passed_task_count']}/{overall['task_count']} "
            f"({_format_rate(overall['task_accuracy'])})"
        ),
        "- Groups are heuristic and non-exclusive.",
        "",
        "| Group | Passed | Tasks | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        item = analysis["groups"][group]
        lines.append(
            f"| {group} | {item['passed_task_count']} | {item['task_count']} | "
            f"{_format_rate(item['task_accuracy'])} |"
        )
    lines.extend(["", "## Language slices", "", "| Language | Passed | Tasks | Accuracy |", "|---|---:|---:|---:|"])
    for language, item in analysis["languages"].items():
        lines.append(
            f"| {language} | {item['passed_task_count']} | {item['task_count']} | "
            f"{_format_rate(item['task_accuracy'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_dataspace_run(
    *,
    run_dir: Path,
    benchmark_root: Path,
    evaluator_script: Path,
    task_ids: list[str] | None = None,
) -> dict[str, Path]:
    evaluation, evaluation_path = run_official_dataspace_evaluator(
        run_dir=run_dir,
        benchmark_root=benchmark_root,
        evaluator_script=evaluator_script,
        task_ids=task_ids,
    )
    _, profiles = profile_dataspace(benchmark_root)
    analysis = build_dataspace_group_analysis(evaluation=evaluation, profiles=profiles)
    analysis_json = run_dir / "dataspace_group_analysis.json"
    analysis_md = run_dir / "dataspace_group_analysis.md"
    analysis_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    analysis_md.write_text(render_dataspace_group_analysis(analysis), encoding="utf-8")
    return {
        "evaluation": evaluation_path,
        "group_analysis_json": analysis_json,
        "group_analysis_md": analysis_md,
    }


def write_dataspace_matrix_comparison(
    *,
    matrix_dir: Path,
    agent_runs: dict[str, Path],
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for agent, run_dir in agent_runs.items():
        evaluation_path = run_dir / "dataspace_evaluation.json"
        group_path = run_dir / "dataspace_group_analysis.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        group_analysis = json.loads(group_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "agent": agent,
                "run_dir": str(run_dir),
                "task_count": evaluation["task_count"],
                "passed_task_count": evaluation["passed_task_count"],
                "task_accuracy": evaluation["task_accuracy"],
                "group_accuracy": {
                    group: group_analysis["groups"][group]["task_accuracy"]
                    for group in GROUP_ORDER
                },
            }
        )

    payload = {
        "metric": "task_accuracy",
        "same_model_protocol": True,
        "groups_are_non_exclusive": True,
        "agents": rows,
    }
    json_path = matrix_dir / "dataspace_matrix_comparison.json"
    md_path = matrix_dir / "dataspace_matrix_comparison.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    header_groups = " | ".join(GROUP_ORDER)
    lines = [
        "# DataSpace Baseline Matrix",
        "",
        f"| Agent | Correct | Overall | {header_groups} |",
        f"|---|---:|---:|{'---:|' * len(GROUP_ORDER)}",
    ]
    for row in rows:
        group_values = " | ".join(
            _format_rate(row["group_accuracy"][group]) for group in GROUP_ORDER
        )
        lines.append(
            f"| {row['agent']} | {row['passed_task_count']}/{row['task_count']} | "
            f"{_format_rate(row['task_accuracy'])} | {group_values} |"
        )
    lines.extend(
        [
            "",
            (
                "All agents use the same model configuration and one worker. Group labels are "
                "heuristic and non-exclusive; the overall score is produced by the official "
                "evaluator."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"comparison_json": json_path, "comparison_md": md_path}
