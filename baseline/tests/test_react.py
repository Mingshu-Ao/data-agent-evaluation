from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.agents.prompt import build_observation_prompt
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


def test_observation_prompt_serializes_date_values() -> None:
    prompt = build_observation_prompt(
        {
            "report_date": date(2026, 7, 30),
            "updated_at": datetime(2026, 7, 30, 12, 34, 56, tzinfo=timezone.utc),
        }
    )

    assert '"report_date": "2026-07-30"' in prompt
    assert '"updated_at": "2026-07-30 12:34:56+00:00"' in prompt


class BudgetAwareModel:
    def complete(self, messages: list[ModelMessage]) -> str:
        last_message = messages[-1].content
        if isinstance(last_message, str) and "Execution budget warning" in last_message:
            return (
                "```json\n"
                '{"thought":"Submit available result.","action":"answer",'
                '"action_input":{"columns":["value"],"rows":[["ready"]]}}\n'
                "```"
            )
        return (
            "```json\n"
            '{"thought":"Inspect context.","action":"list_context",'
            '"action_input":{"max_depth":2}}\n'
            "```"
        )


class InitiallyStubbornModel:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        self.call_count += 1
        if self.call_count < 3:
            return (
                "```json\n"
                '{"thought":"Keep inspecting.","action":"list_context",'
                '"action_input":{"max_depth":2}}\n'
                "```"
            )
        return (
            "```json\n"
            '{"thought":"Submit available result.","action":"answer",'
            '"action_input":{"columns":["value"],"rows":[["ready"]]}}\n'
            "```"
        )


class RecoveryOnlyModel:
    def complete(self, messages: list[ModelMessage]) -> str:
        first_message = messages[0].content
        if isinstance(first_message, str) and "final-answer recovery stage" in first_message:
            return (
                "```json\n"
                '{"thought":"Recover final table.","action":"answer",'
                '"action_input":{"columns":["value"],"rows":[["recovered"]]}}\n'
                "```"
            )
        return (
            "```json\n"
            '{"thought":"Keep inspecting.","action":"list_context",'
            '"action_input":{"max_depth":2}}\n'
            "```"
        )


def test_react_reserves_final_steps_for_answer(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(
            task_id="task_test",
            difficulty="easy",
            question="Return the value.",
        ),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    agent = ReActAgent(
        model=BudgetAwareModel(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=3, answer_reserve_steps=2),
    )

    result = agent.run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["ready"]]
    assert [step.action for step in result.steps] == ["list_context", "answer"]


def test_react_blocks_analysis_tools_inside_answer_reserve(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(
            task_id="task_test",
            difficulty="easy",
            question="Return the value.",
        ),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    agent = ReActAgent(
        model=InitiallyStubbornModel(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4, answer_reserve_steps=3),
    )

    result = agent.run(task)

    assert result.succeeded
    assert result.steps[1].ok is False
    assert "reserved for final submission" in result.steps[1].observation["error"]
    assert result.steps[-1].action == "answer"


def test_react_recovers_answer_after_step_budget_exhaustion(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(
            task_id="task_test",
            difficulty="easy",
            question="Return the value.",
        ),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    agent = ReActAgent(
        model=RecoveryOnlyModel(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=2, answer_reserve_steps=1),
    )

    result = agent.run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [["recovered"]]
    assert result.steps[-1].step_index == 3
    assert result.steps[-1].observation["recovery_stage"] is True
