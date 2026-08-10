from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.agents.ace_playbook import (
    ACEPlaybook,
    curate_ace_playbook_from_run,
)
from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import load_app_config
from data_agent_baseline.tools.registry import create_default_tool_registry


def _task(tmp_path: Path, task_id: str = "task_1") -> PublicTask:
    task_dir = tmp_path / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "patients.csv").write_text(
        "id,status\n1,severe\n2,mild\n",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "question": "List severe patient ids.",
            }
        ),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id=task_id,
            difficulty="unknown",
            question="List severe patient ids.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class CaptureAnswerModel:
    def __init__(self) -> None:
        self.messages: list[ModelMessage] = []

    def complete(self, messages: list[ModelMessage]) -> str:
        self.messages = messages
        return (
            "```json\n"
            '{"thought":"Submit.","action":"answer",'
            '"action_input":{"columns":["id"],"rows":[[1]]}}\n'
            "```"
        )


class InspectForeverModel:
    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        return (
            "```json\n"
            '{"thought":"Inspect.","action":"list_context",'
            '"action_input":{"max_depth":2}}\n'
            "```"
        )


def test_ace_lite_curates_verified_run_without_answers(tmp_path: Path) -> None:
    dataset_root = tmp_path / "input"
    task = _task(dataset_root.parent)
    assert task.assets.task_dir == dataset_root.parent / "task_1"
    dataset_root = dataset_root.parent
    run_dir = tmp_path / "run"
    task_run_dir = run_dir / "task_1"
    task_run_dir.mkdir(parents=True)
    (task_run_dir / "trace.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "succeeded": True,
                "steps": [
                    {"action": "profile_context"},
                    {"action": "execute_structured_sql"},
                    {"action": "answer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluation_path = run_dir / "dataspace_evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "passed": True,
                        "error_code": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    playbook_path = tmp_path / "ace_playbook.json"

    report = curate_ace_playbook_from_run(
        run_dir=run_dir,
        dataset_root=dataset_root,
        evaluation_path=evaluation_path,
        playbook_path=playbook_path,
    )
    entries = ACEPlaybook(playbook_path).entries()

    assert report["delta_count"] == 1
    assert report["entry_count"] == 1
    assert entries[0].kind == "strategy"
    assert "execute_structured_sql" in entries[0].text
    assert "value-1" not in playbook_path.read_text(encoding="utf-8")

    curate_ace_playbook_from_run(
        run_dir=run_dir,
        dataset_root=dataset_root,
        evaluation_path=evaluation_path,
        playbook_path=playbook_path,
    )
    repeated_entries = ACEPlaybook(playbook_path).entries()
    assert repeated_entries[0].helpful_count == 1


def test_react_enhancement_injects_schema_and_ace_playbook(tmp_path: Path) -> None:
    task = _task(tmp_path)
    playbook_path = tmp_path / "ace_playbook.json"
    ACEPlaybook(playbook_path).apply_deltas(
        [
            {
                "entry_id": "strategy:test",
                "kind": "strategy",
                "text": "Use structured SQL for severe patient filtering.",
                "keywords": ["severe", "patient", "csv"],
                "helpful_delta": 2,
                "harmful_delta": 0,
                "evidence_task_ids": ["task_9"],
            }
        ]
    )
    model = CaptureAnswerModel()
    agent = ReActAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(
            max_steps=2,
            schema_index_enabled=True,
            ace_enabled=True,
            ace_playbook_path=playbook_path,
        ),
    )

    result = agent.run(task)
    first_user_message = model.messages[1].content

    assert result.succeeded
    assert isinstance(first_user_message, str)
    assert "Automatically generated schema index" in first_user_message
    assert "patients.csv [csv]: columns=id, status" in first_user_message
    assert "ACE-lite playbook lessons" in first_user_message


def test_react_can_disable_convergence_for_ablation(tmp_path: Path) -> None:
    task = _task(tmp_path)
    agent = ReActAgent(
        model=InspectForeverModel(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=1, convergence_enabled=False),
    )

    result = agent.run(task)

    assert not result.succeeded
    assert len(result.steps) == 1
    assert result.steps[0].action == "list_context"


def test_config_reads_api_key_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DMX_API_KEY", "secret-from-environment")
    config_path = tmp_path / "dataspace.yaml"
    config_path.write_text(
        "agent:\n  api_key: ${DMX_API_KEY}\n",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.agent.api_key == "secret-from-environment"
