from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.dataspace import (
    build_dataspace_group_analysis,
    evaluate_dataspace_run,
    profile_dataspace,
    write_dataspace_report_and_suites,
)
from data_agent_baseline.benchmark.dataspace_download import (
    select_dataspace_archive_members,
)

OFFICIAL_EVALUATOR = (
    Path(__file__).resolve().parents[4]
    / "dataspace-official"
    / "evaluation"
    / "evaluate.py"
)


def _write_csv(path: Path, header: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header])
        writer.writerow([value])


def _write_dataspace_task(
    root: Path,
    *,
    task_id: str,
    question: str,
    files: list[str],
) -> None:
    task_dir = root / "input" / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "question": question}),
        encoding="utf-8",
    )
    for relative_path in files:
        path = context_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


def _write_public_reference(root: Path, task_id: str, value: str) -> None:
    _write_csv(root / "output" / task_id / "gold.csv", "answer", value)
    config_root = root / "evaluation" / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / f"{task_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task_id": task_id,
                "order_sensitive": False,
                "columns": [
                    {
                        "gold_index": 0,
                        "gold_name": "answer",
                        "type": "text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _build_fixture(root: Path) -> None:
    _write_dataspace_task(
        root,
        task_id="task_1",
        question="List the matching records.",
        files=["records.csv"],
    )
    _write_dataspace_task(
        root,
        task_id="task_2",
        question="Join customers and orders.",
        files=[],
    )
    database_path = root / "input" / "task_2" / "context" / "warehouse.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER)")
        connection.execute("CREATE TABLE orders (customer_id INTEGER)")
    _write_dataspace_task(
        root,
        task_id="task_3",
        question="Read the report.",
        files=["report.pdf"],
    )
    _write_dataspace_task(
        root,
        task_id="task_4",
        question="Read the briefing video.",
        files=["briefing.mp4"],
    )
    _write_dataspace_task(
        root,
        task_id="task_5",
        question="Combine the table and PDF evidence.",
        files=["records.csv", "report.pdf"],
    )
    for index in range(1, 6):
        _write_public_reference(root, f"task_{index}", f"value-{index}")


def test_dataspace_profiles_and_public_suites(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "DataSpace-Benchmark"
    _build_fixture(benchmark_root)

    layout, profiles = profile_dataspace(tmp_path)
    profiles_by_id = {profile.task_id: profile for profile in profiles}

    assert layout.benchmark_root == benchmark_root
    assert len(profiles) == 5
    assert profiles_by_id["task_1"].structured
    assert profiles_by_id["task_2"].join
    assert profiles_by_id["task_2"].sqlite_table_count == 2
    assert profiles_by_id["task_3"].has_pdf
    assert profiles_by_id["task_4"].has_video
    assert profiles_by_id["task_5"].mixed_modal
    assert all(profile.public_reference for profile in profiles)

    outputs = write_dataspace_report_and_suites(
        benchmark_root=benchmark_root,
        report_dir=tmp_path / "reports",
        suites_dir=tmp_path / "suites",
        smoke_size=5,
        coverage_size=3,
    )
    smoke = json.loads(outputs["suite_smoke"].read_text(encoding="utf-8"))
    coverage = json.loads(outputs["suite_coverage"].read_text(encoding="utf-8"))
    public = json.loads(outputs["suite_public"].read_text(encoding="utf-8"))

    assert smoke["task_ids"] == [f"task_{index}" for index in range(1, 6)]
    assert len(coverage["task_ids"]) == 3
    assert public["public_summary"]["public_reference_count"] == 5
    assert outputs["report_md"].is_file()


def test_official_evaluator_and_group_analysis(tmp_path: Path) -> None:
    if not OFFICIAL_EVALUATOR.is_file():
        pytest.skip("DataSpace official evaluator is not included in the Phase 1 smoke package.")
    benchmark_root = tmp_path / "DataSpace-Benchmark"
    _build_fixture(benchmark_root)
    run_dir = tmp_path / "run"
    _write_csv(run_dir / "task_1" / "prediction.csv", "different header", "value-1")
    _write_csv(run_dir / "task_2" / "prediction.csv", "answer", "wrong")

    paths = evaluate_dataspace_run(
        run_dir=run_dir,
        benchmark_root=benchmark_root,
        evaluator_script=OFFICIAL_EVALUATOR,
        task_ids=["task_1", "task_2"],
    )
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    group_analysis = json.loads(paths["group_analysis_json"].read_text(encoding="utf-8"))

    assert evaluation["metric"] == "task_accuracy"
    assert evaluation["task_count"] == 2
    assert evaluation["passed_task_count"] == 1
    assert evaluation["task_accuracy"] == 0.5
    assert group_analysis["groups"]["structured"]["task_count"] == 2
    assert group_analysis["groups"]["join"]["task_count"] == 1
    assert paths["group_analysis_md"].is_file()


def test_group_analysis_keeps_empty_groups_explicit(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "DataSpace-Benchmark"
    _build_fixture(benchmark_root)
    _, profiles = profile_dataspace(benchmark_root)
    analysis = build_dataspace_group_analysis(
        evaluation={
            "metric": "task_accuracy",
            "task_count": 1,
            "passed_task_count": 1,
            "task_accuracy": 1.0,
            "tasks": [{"task_id": "task_1", "passed": True}],
        },
        profiles=profiles,
    )

    assert analysis["groups"]["structured"]["task_accuracy"] == 1.0
    assert analysis["groups"]["video"]["task_count"] == 0
    assert analysis["groups"]["video"]["task_accuracy"] is None


def test_dataspace_subset_selects_only_complete_requested_tasks() -> None:
    root = "DataSpace-Benchmark"
    members = [
        f"{root}/README.md",
        f"{root}/input/task_10/task.json",
        f"{root}/input/task_10/context/data.csv",
        f"{root}/output/task_10/gold.csv",
        f"{root}/evaluation/configs/task_10.json",
        f"{root}/input/task_11/task.json",
        f"{root}/input/task_11/context/other.csv",
    ]

    selected = select_dataspace_archive_members(members, task_ids=["task_10"])

    assert f"{root}/input/task_10/context/data.csv" in selected
    assert f"{root}/output/task_10/gold.csv" in selected
    assert not any("task_11" in name for name in selected)
