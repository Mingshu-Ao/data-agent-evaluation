from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from data_agent_baseline.agents.multimodal import (
    build_user_content_with_video,
    find_attached_keyframes,
)
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.fdabench import (
    evaluate_fdabench_replay,
    load_fdabench,
    materialize_fdabench_multiple_replay,
    summarize_fdabench,
    write_fdabench_report_and_suite,
)
from data_agent_baseline.benchmark.phase_comparison import compare_kdd_phases
from data_agent_baseline.benchmark.phase_run_comparison import compare_phase_runs
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.benchmark.step_limit_report import (
    diagnose_step_limit_trace,
)
from data_agent_baseline.benchmark.suites import (
    build_suite_payload,
    infer_question_type,
)
from data_agent_baseline.benchmark.video_preprocessing import (
    render_video_evidence,
    sample_video_timestamps,
    select_visual_change_timestamps,
    visual_change_score,
)


def _write_kdd_task(
    root: Path,
    *,
    task_id: str,
    question: str,
    difficulty: str | None,
    files: list[str],
) -> None:
    task_dir = root / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    payload = {"task_id": task_id, "question": question}
    if difficulty is not None:
        payload["difficulty"] = difficulty
    (task_dir / "task.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    for relative_path in files:
        path = context_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def _write_gold(root: Path, task_id: str, rows: list[list[str]]) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    with (task_dir / "gold.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer"])
        writer.writerows(rows)


def _write_run_task(
    run_root: Path,
    *,
    task_id: str,
    succeeded: bool,
    rows: list[list[str]] | None,
    failure_reason: str | None = None,
) -> None:
    task_dir = run_root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "trace.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "succeeded": succeeded,
                "failure_reason": failure_reason,
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    if rows is not None:
        with (task_dir / "prediction.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["answer"])
            writer.writerows(rows)


def test_phase2_task_without_difficulty_and_video_filter(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_kdd_task(
        input_root,
        task_id="task_1",
        question="统计满足条件的公司数量。",
        difficulty=None,
        files=["data.csv", "briefing.mp4", ".DS_Store"],
    )
    _write_kdd_task(
        input_root,
        task_id="task_2",
        question="列出所有非空记录。",
        difficulty=None,
        files=["data.csv", "knowledge.md"],
    )
    dataset = DABenchPublicDataset(input_root)

    assert dataset.get_task("task_1").difficulty == "unknown"
    payload = build_suite_payload(
        dataset=dataset,
        suite_name="text_only",
        suite_size=10,
        description="fixture",
        exclude_file_types={"mp4"},
    )

    assert payload["task_ids"] == ["task_2"]
    assert payload["dataset_summary"]["multimodal_task_count"] == 1
    assert infer_question_type("统计满足条件的公司数量。") == "count"


def test_video_timestamp_sampling_and_evidence_rendering() -> None:
    timestamps = sample_video_timestamps(
        88.5,
        interval_seconds=10.0,
        max_frames=12,
    )
    evidence = render_video_evidence(
        source_video="briefing.mp4",
        duration_seconds=88.5,
        audio_stream_count=1,
        frames=[
            {
                "frame_index": 1,
                "timestamp_seconds": timestamps[0],
                "image_path": "video_evidence/frames/frame_001.jpg",
                "ocr_lines": [
                    {
                        "text": "Threshold: 100",
                        "confidence": 0.99,
                        "visually_highlighted": True,
                    },
                ],
            }
        ],
    )

    assert timestamps[0] == 4.425
    assert timestamps[-1] == 84.425
    assert len(timestamps) == 9
    assert "Threshold: 100" in evidence
    assert "LIKELY SELECTED" in evidence
    assert "Audio transcription: not generated" in evidence


def test_visual_change_scoring_and_timestamp_selection() -> None:
    unchanged = np.zeros((8, 8), dtype=np.uint8)
    changed = np.full((8, 8), 255, dtype=np.uint8)

    assert visual_change_score(unchanged, unchanged) == 0.0
    assert visual_change_score(unchanged, changed) == 1.0

    timestamps = select_visual_change_timestamps(
        [
            (0.0, 0.0),
            (1.0, 0.02),
            (3.0, 0.75),
            (4.0, 0.20),
            (8.0, 0.60),
        ],
        duration_seconds=10.0,
        interval_seconds=5.0,
        max_frames=3,
        minimum_change_score=0.10,
        minimum_gap_seconds=2.0,
    )

    assert timestamps == [0.0, 3.0, 8.0]

    coverage_timestamps = select_visual_change_timestamps(
        [(float(index), 0.01) for index in range(20)],
        duration_seconds=20.0,
        interval_seconds=4.0,
        max_frames=6,
        minimum_change_score=0.10,
        minimum_gap_seconds=2.0,
    )
    assert len(coverage_timestamps) == 6
    assert coverage_timestamps[0] == 0.0
    assert coverage_timestamps[-1] == 19.0


def test_asr_evidence_and_keyframe_attachment_marker(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    frame_dir = context_dir / "video_evidence" / "frames"
    frame_dir.mkdir(parents=True)
    frame_path = frame_dir / "frame_001.jpg"
    frame_path.write_bytes(b"jpeg fixture")
    manifest = {
        "attach_keyframes": True,
        "frames": [
            {
                "image_path": "video_evidence/frames/frame_001.jpg",
            }
        ],
    }
    (context_dir / "video_evidence" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    evidence = render_video_evidence(
        source_video="briefing.mp4",
        duration_seconds=10.0,
        audio_stream_count=1,
        sampling_strategy="visual_change",
        frames=[],
        transcript={
            "language": "zh",
            "language_probability": 0.98,
            "segments": [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 2.5,
                    "text": "选择二零一九年",
                }
            ],
        },
    )

    assert find_attached_keyframes(context_dir) == [frame_path]
    assert "Frame sampling strategy: visual_change" in evidence
    assert "ASR Transcript" in evidence
    assert "选择二零一九年" in evidence

    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="unknown", question="Question"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    user_content = build_user_content_with_video(task, "Question")
    assert isinstance(user_content, list)
    assert [item["type"] for item in user_content] == ["text", "image_url"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_phase_comparison_reports_new_modalities(tmp_path: Path) -> None:
    phase1_input = tmp_path / "phase1" / "input"
    phase1_gold = tmp_path / "phase1" / "output"
    phase2_input = tmp_path / "phase2" / "input"
    phase2_gold = tmp_path / "phase2" / "output"
    _write_kdd_task(
        phase1_input,
        task_id="task_1",
        question="List records.",
        difficulty="easy",
        files=["data.csv"],
    )
    _write_gold(phase1_gold, "task_1", [["1"]])
    _write_kdd_task(
        phase2_input,
        task_id="task_1",
        question="列出记录。",
        difficulty=None,
        files=["document.pdf", "video.mp4"],
    )
    _write_gold(phase2_gold, "task_1", [["1"], ["2"]])

    report = compare_kdd_phases(
        phase1_input=phase1_input,
        phase1_gold=phase1_gold,
        phase2_input=phase2_input,
        phase2_gold=phase2_gold,
    )

    assert report["phase2"]["multimodal_task_count"] == 1
    assert report["comparison"]["new_phase2_file_types"] == ["mp4", "pdf"]


def test_phase_run_comparison_separates_success_and_accuracy(tmp_path: Path) -> None:
    phase1_run = tmp_path / "phase1_run"
    phase1_gold = tmp_path / "phase1_gold"
    phase2_run = tmp_path / "phase2_run"
    phase2_gold = tmp_path / "phase2_gold"
    _write_gold(phase1_gold, "task_1", [["1"]])
    _write_gold(phase1_gold, "task_2", [["2"]])
    _write_run_task(phase1_run, task_id="task_1", succeeded=True, rows=[["1"]])
    _write_run_task(phase1_run, task_id="task_2", succeeded=True, rows=[["wrong"]])
    _write_gold(phase2_gold, "task_1", [["1"]])
    _write_gold(phase2_gold, "task_2", [["2"]])
    _write_run_task(phase2_run, task_id="task_1", succeeded=True, rows=[["1"]])
    _write_run_task(
        phase2_run,
        task_id="task_2",
        succeeded=False,
        rows=None,
        failure_reason="Agent did not submit an answer within max_steps.",
    )

    result = compare_phase_runs(
        phase1_run_dir=phase1_run,
        phase1_gold_root=phase1_gold,
        phase2_run_dir=phase2_run,
        phase2_gold_root=phase2_gold,
    )

    assert result["phase1"]["run_success_rate"] == 1.0
    assert result["phase1"]["exact_match_rate"] == 0.5
    assert result["phase2"]["run_success_rate"] == 0.5
    assert result["phase2"]["failure_counts"] == {"max_steps": 1}
    assert result["rate_delta_phase2_minus_phase1"]["run_success_rate"] == -0.5


def test_step_limit_diagnosis_finds_repeated_action_loop() -> None:
    repeated_step = {
        "action": "read_doc",
        "action_input": {"path": "knowledge.md"},
        "ok": True,
    }
    trace = {
        "task_id": "task_1",
        "failure_reason": "Agent did not submit an answer within max_steps.",
        "steps": [repeated_step, repeated_step, repeated_step, repeated_step],
    }

    diagnosis = diagnose_step_limit_trace(trace)

    assert diagnosis["category"] == "repeated_action_loop"
    assert diagnosis["max_consecutive_repeat"] == 4


def _fdabench_payload(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "instance_id": f"instance_{task_id}",
        "db": "fixture_db",
        "level": "easy",
        "database_type": "bird",
        "question_type": "report",
        "tools_available": ["execute_sql", "web_search", "vector_search"],
        "query": "Analyze the fixture.",
    }


def test_fdabench_loader_and_suite(tmp_path: Path) -> None:
    for split in ("report", "single", "multiple"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        payload = _fdabench_payload(f"{split}_1")
        if split == "report":
            payload.update(
                {
                    "ground_truth_report": "Reference report.",
                    "sql_result": "[[1]]",
                    "frozen_web_search": {"searches": []},
                    "frozen_vector_search": {"searches": []},
                }
            )
        if split == "multiple":
            payload["correct_answer"] = ["A", "B"]
        (split_dir / "data.jsonl").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    records = load_fdabench(tmp_path)
    summary = summarize_fdabench(records)
    _, _, suite_path = write_fdabench_report_and_suite(
        root=tmp_path,
        output_dir=tmp_path / "reports",
        suite_size=3,
    )

    assert summary["task_count"] == 3
    assert summary["released_answer_count"] == 1
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert suite["suite_summary"]["split_counts"] == {
        "multiple": 1,
        "report": 1,
        "single": 1,
    }

    replay_input, replay_gold, manifest_path = materialize_fdabench_multiple_replay(
        root=tmp_path,
        output_root=tmp_path / "replay",
        size=1,
    )
    replay_dataset = DABenchPublicDataset(replay_input)
    assert replay_dataset.list_task_ids() == ["task_1"]
    replay_task = replay_dataset.get_task("task_1")
    assert "multiple-select" in replay_task.question
    assert "return all supported correct option labels" in replay_task.question
    assert (replay_gold / "task_1" / "gold.csv").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "frozen_context_replay"

    run_dir = tmp_path / "run"
    prediction_dir = run_dir / "task_1"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "prediction.csv").write_text(
        "answer\nA\n",
        encoding="utf-8",
    )
    evaluation = evaluate_fdabench_replay(
        run_dir=run_dir,
        gold_root=replay_gold,
    )
    assert evaluation["exact_match_count"] == 0
    assert evaluation["micro_option_precision"] == 1.0
    assert evaluation["micro_option_recall"] == 0.5
    assert evaluation["micro_option_f1"] == 0.666667
