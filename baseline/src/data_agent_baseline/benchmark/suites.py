from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask


@dataclass(frozen=True, slots=True)
class TaskProfile:
    task_id: str
    difficulty: str
    file_types: tuple[str, ...]
    file_type_set: str
    question_type: str
    question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "file_types": list(self.file_types),
            "file_type_set": self.file_type_set,
            "question_type": self.question_type,
            "question": self.question,
        }


def infer_question_type(question: str) -> str:
    text = question.lower()
    if any(token in text for token in ("how many", "count", "number of", "多少", "数目", "数量", "统计")):
        return "count"
    if any(token in text for token in ("average", "avg", "mean", "平均")):
        return "average"
    if any(
        token in text
        for token in (
            "lowest",
            "highest",
            "best",
            "worst",
            "maximum",
            "minimum",
            "most",
            "least",
            "最高",
            "最低",
            "最大",
            "最小",
            "最多",
            "最少",
        )
    ):
        return "ranking_or_extreme"
    if any(token in text for token in ("total", "sum", "总计", "合计", "总额", "求和")):
        return "sum"
    if any(token in text for token in ("date", "when", "日期", "时间", "何时")):
        return "date_lookup"
    if any(
        token in text
        for token in ("list", "provide", "state", "which", "what", "列出", "哪些", "哪个", "展示")
    ):
        return "lookup_or_list"
    return "other"


def profile_task(task: PublicTask) -> TaskProfile:
    suffixes: set[str] = set()
    for path in task.context_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.lower() in {"thumbs.db", "desktop.ini"}:
            continue
        suffix = path.suffix.lower().lstrip(".") or "other"
        suffixes.add(suffix)
    file_types = tuple(sorted(suffixes))
    return TaskProfile(
        task_id=task.task_id,
        difficulty=task.difficulty,
        file_types=file_types,
        file_type_set="+".join(file_types),
        question_type=infer_question_type(task.question),
        question=task.question,
    )


def profile_dataset(dataset: DABenchPublicDataset) -> list[TaskProfile]:
    return [profile_task(task) for task in dataset.iter_tasks()]


def _target_difficulty_counts(profiles: list[TaskProfile], suite_size: int) -> dict[str, int]:
    total = len(profiles)
    raw_counts = Counter(profile.difficulty for profile in profiles)
    targets = {
        difficulty: max(1, round(count / total * suite_size))
        for difficulty, count in raw_counts.items()
    }
    while sum(targets.values()) > suite_size:
        difficulty = max(targets, key=lambda item: targets[item])
        targets[difficulty] -= 1
    while sum(targets.values()) < suite_size:
        difficulty = max(raw_counts, key=lambda item: raw_counts[item] - targets.get(item, 0))
        targets[difficulty] = targets.get(difficulty, 0) + 1
    return targets


def select_coverage_suite(profiles: list[TaskProfile], *, suite_size: int) -> list[TaskProfile]:
    if suite_size <= 0:
        raise ValueError("suite_size must be positive.")
    if suite_size >= len(profiles):
        return list(profiles)

    selected: list[TaskProfile] = []
    selected_ids: set[str] = set()
    targets = _target_difficulty_counts(profiles, suite_size)
    selected_difficulty_counts: Counter[str] = Counter()

    def add(profile: TaskProfile) -> bool:
        if profile.task_id in selected_ids or len(selected) >= suite_size:
            return False
        selected.append(profile)
        selected_ids.add(profile.task_id)
        selected_difficulty_counts[profile.difficulty] += 1
        return True

    by_file_type: dict[str, list[TaskProfile]] = defaultdict(list)
    by_question_type: dict[str, list[TaskProfile]] = defaultdict(list)
    for profile in profiles:
        by_file_type[profile.file_type_set].append(profile)
        by_question_type[profile.question_type].append(profile)

    for group in (by_file_type, by_question_type):
        for _, candidates in sorted(group.items(), key=lambda item: (-len(item[1]), item[0])):
            candidates = sorted(
                candidates,
                key=lambda profile: (
                    selected_difficulty_counts[profile.difficulty] >= targets.get(profile.difficulty, 0),
                    int(profile.task_id.removeprefix("task_")),
                ),
            )
            for candidate in candidates:
                if add(candidate):
                    break

    remaining = sorted(
        profiles,
        key=lambda profile: (
            selected_difficulty_counts[profile.difficulty] >= targets.get(profile.difficulty, 0),
            profile.file_type_set in {item.file_type_set for item in selected},
            profile.question_type in {item.question_type for item in selected},
            int(profile.task_id.removeprefix("task_")),
        ),
    )
    for profile in remaining:
        add(profile)
        if len(selected) >= suite_size:
            break

    selected.sort(key=lambda profile: int(profile.task_id.removeprefix("task_")))
    return selected


def summarize_profiles(profiles: list[TaskProfile]) -> dict[str, Any]:
    file_type_counts: Counter[str] = Counter()
    for profile in profiles:
        file_type_counts.update(profile.file_types)
    return {
        "task_count": len(profiles),
        "difficulty_counts": dict(sorted(Counter(profile.difficulty for profile in profiles).items())),
        "file_type_counts": dict(sorted(file_type_counts.items())),
        "file_type_set_counts": dict(sorted(Counter(profile.file_type_set for profile in profiles).items())),
        "question_type_counts": dict(sorted(Counter(profile.question_type for profile in profiles).items())),
        "multimodal_task_count": sum(
            any(file_type in {"avi", "m4v", "mkv", "mov", "mp4", "webm"} for file_type in profile.file_types)
            for profile in profiles
        ),
    }


def build_suite_payload(
    *,
    dataset: DABenchPublicDataset,
    suite_name: str,
    suite_size: int,
    description: str,
    require_file_types: set[str] | None = None,
    exclude_file_types: set[str] | None = None,
) -> dict[str, Any]:
    all_profiles = profile_dataset(dataset)
    required = {value.lower().lstrip(".") for value in (require_file_types or set())}
    excluded = {value.lower().lstrip(".") for value in (exclude_file_types or set())}
    profiles = [
        profile
        for profile in all_profiles
        if required.issubset(set(profile.file_types))
        and not excluded.intersection(profile.file_types)
    ]
    if not profiles:
        raise ValueError("No tasks match the requested file-type filters.")
    selected = select_coverage_suite(profiles, suite_size=suite_size)
    return {
        "suite_name": suite_name,
        "description": description,
        "selection_policy": (
            "Greedy coverage over file-type sets and question types, then balanced by difficulty. "
            "Use the full suite when suite_size is greater than or equal to the dataset size."
        ),
        "dataset_summary": summarize_profiles(all_profiles),
        "eligible_summary": summarize_profiles(profiles),
        "filters": {
            "require_file_types": sorted(required),
            "exclude_file_types": sorted(excluded),
        },
        "suite_summary": summarize_profiles(selected),
        "task_ids": [profile.task_id for profile in selected],
        "tasks": [profile.to_dict() for profile in selected],
    }


def write_suite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_suite_task_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_task_ids = payload.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(isinstance(item, str) for item in raw_task_ids):
        raise ValueError(f"Suite file must contain a string list field `task_ids`: {path}")
    return list(raw_task_ids)
