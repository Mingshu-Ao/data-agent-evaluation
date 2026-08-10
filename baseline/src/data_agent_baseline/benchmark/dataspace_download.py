from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

DATASPACE_ARCHIVE_URL = (
    "https://huggingface.co/datasets/HKUSTDial/DataSpace/resolve/main/"
    "release/DataSpace-Benchmark.zip?download=true"
)
DATASPACE_ARCHIVE_ROOT = "DataSpace-Benchmark"

# About 30 MB of uncompressed context in the current release. All five tasks
# have public references and jointly cover PDF, SQLite, video, and mixed-modal workspaces.
DEFAULT_LOCAL_SMOKE_TASK_IDS = (
    "task_35",
    "task_58",
    "task_312",
    "task_328",
    "task_391",
)


def _task_number(task_id: str) -> int:
    if not task_id.startswith("task_"):
        raise ValueError(f"Invalid DataSpace task id: {task_id}")
    return int(task_id.removeprefix("task_"))


def normalize_task_ids(task_ids: Iterable[str]) -> tuple[str, ...]:
    normalized = sorted({value.strip() for value in task_ids if value.strip()}, key=_task_number)
    if not normalized:
        raise ValueError("At least one DataSpace task id is required.")
    return tuple(normalized)


def select_dataspace_archive_members(
    member_names: Iterable[str],
    *,
    task_ids: Iterable[str],
) -> list[str]:
    selected_ids = normalize_task_ids(task_ids)
    prefixes = tuple(
        f"{DATASPACE_ARCHIVE_ROOT}/input/{task_id}/" for task_id in selected_ids
    ) + tuple(
        f"{DATASPACE_ARCHIVE_ROOT}/output/{task_id}/" for task_id in selected_ids
    )
    config_names = {
        f"{DATASPACE_ARCHIVE_ROOT}/evaluation/configs/{task_id}.json"
        for task_id in selected_ids
    }
    shared_names = {
        f"{DATASPACE_ARCHIVE_ROOT}/CITATION.cff",
        f"{DATASPACE_ARCHIVE_ROOT}/LICENSE",
        f"{DATASPACE_ARCHIVE_ROOT}/README.md",
        f"{DATASPACE_ARCHIVE_ROOT}/evaluation/README.md",
    }
    selected = [
        name
        for name in member_names
        if name in shared_names
        or name in config_names
        or any(name.startswith(prefix) for prefix in prefixes)
    ]
    available = set(selected)
    for task_id in selected_ids:
        task_json = f"{DATASPACE_ARCHIVE_ROOT}/input/{task_id}/task.json"
        gold_csv = f"{DATASPACE_ARCHIVE_ROOT}/output/{task_id}/gold.csv"
        config_json = f"{DATASPACE_ARCHIVE_ROOT}/evaluation/configs/{task_id}.json"
        missing = [
            name for name in (task_json, gold_csv, config_json) if name not in available
        ]
        if missing:
            raise ValueError(
                f"Task {task_id} is not a complete public-reference task in the archive: "
                f"{', '.join(missing)}"
            )
    return sorted(selected)


def _safe_destination(output_root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    destination = output_root.joinpath(*relative.parts).resolve()
    resolved_root = output_root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"ZIP member escapes output root: {member_name}")
    return destination


def download_dataspace_subset(
    *,
    output_root: Path,
    task_ids: Iterable[str] = DEFAULT_LOCAL_SMOKE_TASK_IDS,
    archive_url: str = DATASPACE_ARCHIVE_URL,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> dict[str, Any]:
    try:
        from remotezip import RemoteZip
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DataSpace subset download requires `remotezip`. Install with "
            "`python -m pip install -e .[dataspace]`."
        ) from exc

    normalized_ids = normalize_task_ids(task_ids)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    downloaded_files = 0
    reused_files = 0
    downloaded_bytes = 0

    with RemoteZip(archive_url) as archive:
        infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        selected_names = select_dataspace_archive_members(
            infos,
            task_ids=normalized_ids,
        )
        total = len(selected_names)
        for index, member_name in enumerate(selected_names, start=1):
            info = infos[member_name]
            destination = _safe_destination(output_root, member_name)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                reused_files += 1
                if progress_callback is not None:
                    progress_callback(index, total, member_name, True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial_path = destination.with_name(destination.name + ".part")
            with archive.open(info) as source, partial_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if partial_path.stat().st_size != info.file_size:
                raise OSError(
                    f"Incomplete ZIP member {member_name}: expected {info.file_size}, "
                    f"received {partial_path.stat().st_size} bytes."
                )
            partial_path.replace(destination)
            downloaded_files += 1
            downloaded_bytes += info.file_size
            if progress_callback is not None:
                progress_callback(index, total, member_name, False)

    benchmark_root = output_root / DATASPACE_ARCHIVE_ROOT
    manifest = {
        "source": archive_url,
        "method": "HTTP Range extraction from the official DataSpace ZIP",
        "task_ids": list(normalized_ids),
        "task_count": len(normalized_ids),
        "downloaded_files": downloaded_files,
        "reused_files": reused_files,
        "downloaded_uncompressed_bytes": downloaded_bytes,
        "benchmark_root": str(benchmark_root),
    }
    manifest_path = output_root / "dataspace_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}
