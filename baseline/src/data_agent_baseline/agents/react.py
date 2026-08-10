from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.agents.ace_playbook import ACEPlaybook
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.multimodal import build_user_content_with_video
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_answer_budget_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.context_profile import build_lightweight_schema_index
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    answer_reserve_steps: int = 4
    schema_index_enabled: bool = False
    convergence_enabled: bool = True
    ace_enabled: bool = False
    ace_playbook_path: Path | None = None


def _render_schema_index(profile: dict[str, object]) -> str:
    lines = [
        "Automatically generated schema index:",
        "Use this only for source discovery; inspect or query the files before answering.",
    ]
    raw_files = profile.get("files", [])
    if not isinstance(raw_files, list):
        return "\n".join(lines)
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "unknown"))
        kind = str(item.get("kind", "file"))
        if kind == "sqlite":
            tables = item.get("tables", [])
            table_parts = []
            if isinstance(tables, list):
                for table in tables:
                    if isinstance(table, dict):
                        table_parts.append(
                            f"{table.get('name')}({', '.join(str(value) for value in table.get('columns', []))})"
                        )
            lines.append(f"- {path} [sqlite]: {'; '.join(table_parts) or 'schema unavailable'}")
        elif kind == "csv":
            columns = ", ".join(str(value) for value in item.get("columns", []))
            lines.append(f"- {path} [csv]: columns={columns}; rows={item.get('row_count')}")
        elif kind == "json":
            keys = item.get("keys", [])
            lines.append(f"- {path} [json]: top-level keys={', '.join(str(value) for value in keys)}")
        elif kind == "document":
            lines.append(f"- {path} [document]: bytes={item.get('size')}")
        else:
            lines.append(f"- {path} [{kind}]")
    raw_errors = profile.get("errors", [])
    if isinstance(raw_errors, list) and raw_errors:
        lines.append(f"- Indexing warnings: {len(raw_errors)} file(s) could not be profiled.")
    return "\n".join(lines)


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    unclosed_fence_match = re.search(r"```(?:json)?\s*", text, flags=re.IGNORECASE)
    if unclosed_fence_match is not None:
        return text[unclosed_fence_match.end() :].strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        candidate = text[match.start() :]
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return candidate[:end].strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise TypeError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise TypeError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise TypeError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.ace_playbook = ACEPlaybook(self.config.ace_playbook_path)
        self._initial_context_cache: dict[str, str] = {}

    def _initial_context(self, task: PublicTask) -> str:
        cached = self._initial_context_cache.get(task.task_id)
        if cached is not None:
            return cached
        sections: list[str] = []
        schema_index = ""
        if self.config.schema_index_enabled:
            try:
                schema_index = _render_schema_index(build_lightweight_schema_index(task))
                sections.append(schema_index)
            except Exception as exc:  # noqa: BLE001
                sections.append(f"Schema index unavailable: {exc}")
        if self.config.ace_enabled:
            playbook = self.ace_playbook.render_for_prompt(
                f"{task.question}\n{schema_index}",
            )
            if playbook:
                sections.append(playbook)
        rendered = "\n\n".join(sections)
        self._initial_context_cache[task.task_id] = rendered
        return rendered

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        remaining_steps: int,
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        task_prompt = build_task_prompt(task)
        initial_context = self._initial_context(task)
        if initial_context:
            task_prompt = f"{task_prompt}\n\n{initial_context}"
        messages.append(
            ModelMessage(
                role="user",
                content=build_user_content_with_video(task, task_prompt),
            )
        )
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        if self.config.convergence_enabled and remaining_steps <= self.config.answer_reserve_steps:
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_answer_budget_prompt(remaining_steps),
                )
            )
        return messages

    def _attempt_forced_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
    ) -> None:
        evidence_items: list[str] = []
        evidence_chars = 0
        for step in reversed(state.steps):
            if not step.ok:
                continue
            rendered = json.dumps(step.observation, ensure_ascii=False, default=str)
            rendered = rendered[:3000]
            if evidence_chars + len(rendered) > 12000:
                break
            evidence_items.append(
                f"Tool `{step.action}` completed with observation:\n{rendered}"
            )
            evidence_chars += len(rendered)
        evidence_items.reverse()
        evidence = "\n\n".join(evidence_items) or "No successful tool evidence is available."
        messages = [
            ModelMessage(
                role="system",
                content=(
                    "You are the final-answer recovery stage of a data agent. Analysis is "
                    "finished and no tools are available except `answer`. Return exactly one "
                    "```json fenced object with `thought`, `action`, and `action_input`. "
                    "The action must be `answer`; action_input must contain `columns` and "
                    "`rows`. Do not request more data or output prose."
                ),
            ),
            ModelMessage(
                role="user",
                content=(
                    f"Question: {task.question}\n\n"
                    f"Collected evidence:\n{evidence}\n\n"
                    "Submit the best table supported by this evidence now. Preserve observed "
                    "source field names and capitalization when possible."
                ),
            ),
        ]
        raw_response = self.model.complete(messages)
        model_step = parse_model_step(raw_response)
        if model_step.action != "answer":
            raise ValueError("Forced final-answer stage must call `answer`.")
        tool_result = self.tools.execute(task, "answer", model_step.action_input)
        state.steps.append(
            StepRecord(
                step_index=self.config.max_steps + 1,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw_response,
                observation={
                    "ok": tool_result.ok,
                    "tool": "answer",
                    "content": tool_result.content,
                    "recovery_stage": True,
                },
                ok=tool_result.ok,
            )
        )
        if tool_result.is_terminal:
            state.answer = tool_result.answer

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            remaining_steps = self.config.max_steps - step_index + 1
            raw_response = self.model.complete(
                self._build_messages(
                    task,
                    state,
                    remaining_steps=remaining_steps,
                )
            )
            try:
                model_step = parse_model_step(raw_response)
                if (
                    self.config.convergence_enabled
                    and
                    remaining_steps <= self.config.answer_reserve_steps
                    and model_step.action != "answer"
                ):
                    observation = {
                        "ok": False,
                        "tool": model_step.action,
                        "error": (
                            "Execution budget is reserved for final submission. "
                            "Do not call another analysis tool; call `answer` now with "
                            "the best table supported by existing observations."
                        ),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:  # noqa: BLE001
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and self.config.convergence_enabled:
            try:
                self._attempt_forced_answer(task, state)
            except Exception as exc:  # noqa: BLE001
                state.steps.append(
                    StepRecord(
                        step_index=self.config.max_steps + 1,
                        thought="",
                        action="__forced_answer_error__",
                        action_input={},
                        raw_response="",
                        observation={"ok": False, "error": str(exc)},
                        ok=False,
                    )
                )
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
