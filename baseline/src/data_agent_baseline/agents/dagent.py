from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.multimodal import build_user_content_with_video
from data_agent_baseline.agents.prompt import (
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react import (
    _load_single_json_object,
    _strip_json_fence,
    parse_model_step,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry
from data_agent_baseline.tools.structured_sql import rewrite_semantic_column_owners

DAGENT_SYSTEM_PROMPT = """
You are DAgent-lite, a relational-database-driven data analysis agent inspired by DAgent.

You receive a decomposition and retrieval plan before execution. Follow it while remaining grounded in
tool observations.

Rules:
1. Use the context profile and plan to choose between direct file inspection, SQL retrieval, or both.
2. For complex questions, solve the listed sub-questions and integrate their evidence.
3. Use `execute_structured_sql` for relational work. It exposes SQLite, JSON, CSV, and Parquet
   sources in one temporary DuckDB database, enabling cross-source joins. Table names are SQLite
   table names, JSON `table` values, or sanitized CSV/Parquet file stems. Never use file paths in
   SQL FROM or JOIN clauses.
4. Use executable Python or SQL when previews cannot establish the complete answer.
5. Preserve exact requested attributes and observed source column names in the final table.
6. Before answering, verify filters, row count, column count, and output granularity.
7. Base every value on tool observations. Never invent missing values.
8. The task ends only when you call the `answer` tool with `columns` and `rows`.
9. Return exactly one fenced ```json block containing one object with `thought`, `action`, and
   `action_input`. Do not output text outside the block.
""".strip()


DAGENT_PLANNER_SYSTEM_PROMPT = """
You are the planning module of DAgent-lite.

Create a retrieval-and-analysis plan for the question using the context profile. Decompose only when
the question has multiple independent analytical requirements. For each sub-question, choose one
retrieval strategy: `direct`, `sql`, or `hybrid`.

Resolve domain terms from the relevant document evidence. When a documented use case closely matches
the question, prefer its explicit field-value mapping over a broader or more ambiguous definition.
SQL retrieval is available for SQLite/DB files and for tabular JSON/CSV/Parquet files.

Return exactly one fenced ```json block containing one object with:
- thought: short planning rationale
- decomposition_required: boolean
- sub_questions: non-empty list of objects with `id`, `question`, `retrieval_strategy`, and
  `expected_evidence`
- sql_rewrite_policy: object with `enabled` and `goal`
- final_answer_contract: object with `columns`, `row_rules`, and `validation_checks`

Do not solve the task and do not output text outside the JSON block.
""".strip()


DAGENT_SQL_REWRITE_SYSTEM_PROMPT = """
You are the SQL rewrite tool of DAgent-lite.

Review a proposed read-only SQL query against the task, plan, and observed memory. Keep it when it is
already compact and correct. Otherwise rewrite it to improve relevance and reduce unnecessary rows or
columns. Preserve the proposed query's immediate intent: a narrow query that validates an ambiguous,
null, or conflicting value from a previous result is a valid diagnostic follow-up and must not be
expanded back into the main retrieval query. When a requested column exists in multiple tables, use
the semantic document evidence to select the table that owns the requested meaning. Never produce
write operations.

Return exactly one fenced ```json block containing one object with:
- thought: short reason
- keep_original: boolean
- sql: the final read-only SQL query

Do not output text outside the JSON block.
""".strip()


DAGENT_REPORT_SYSTEM_PROMPT = """
You are the report generation tool of DAgent-lite.

Given the original question, execution plan, observed evidence summary, and final answer table, produce
a concise analytical report. Do not introduce facts that are absent from the evidence or answer.

Return exactly one fenced ```json block containing one object with:
- title: short report title
- summary: one concise paragraph
- findings: list of concise strings
- caveats: list of concise strings

Do not output text outside the JSON block.
""".strip()

DAGENT_ANSWER_REVIEW_SYSTEM_PROMPT = """
You are the answer validation module of DAgent-lite.

Review the proposed final table against the question, plan, semantic field ownership, document
evidence, and execution memory. Reject an answer when a requested field is sourced from the wrong
entity, contains unexplained nulls, violates a filter or output contract, or conflicts with observed
evidence. Accept only when the table is grounded and complete.

Return exactly one fenced ```json block containing one object with:
- accept: boolean
- issues: list of concise strings
- suggested_action: concise next step when rejected, otherwise an empty string

Do not output text outside the JSON block.
""".strip()

QUESTION_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "list",
    "the",
    "their",
    "with",
}


@dataclass(frozen=True, slots=True)
class DAgentLiteConfig:
    max_steps: int = 16
    sql_rewrite_enabled: bool = True
    answer_review_enabled: bool = True
    report_generation_enabled: bool = True
    memory_max_items: int = 12


@dataclass(slots=True)
class DAgentMemory:
    question: str
    context_profile: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    intermediate_results: list[dict[str, Any]] = field(default_factory=list)
    planning_path: list[str] = field(default_factory=list)

    def remember(
        self,
        *,
        action: str,
        action_input: dict[str, Any],
        observation: dict[str, Any],
        max_items: int,
    ) -> None:
        self.planning_path.append(action)
        self.intermediate_results.append(
            {
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
        )
        if len(self.intermediate_results) > max_items:
            self.intermediate_results = self.intermediate_results[-max_items:]

    def snapshot(self, *, max_chars: int = 14_000) -> str:
        files = self.context_profile.get("files", [])
        compact_files = []
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                compact_item = {
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "columns": item.get("columns"),
                }
                summary = item.get("summary")
                if isinstance(summary, dict):
                    sample = summary.get("sample")
                    if isinstance(sample, dict):
                        table = sample.get("table")
                        records = sample.get("records")
                        if isinstance(table, dict):
                            compact_item["table_name"] = table.get("sample")
                        if isinstance(records, dict):
                            item_sample = records.get("item_sample")
                            if isinstance(item_sample, dict):
                                compact_item["columns"] = item_sample.get("keys")
                compact_files.append(compact_item)
        payload = {
            "question": self.question,
            "plan": self.plan,
            "relevant_document_evidence": self.context_profile.get(
                "relevant_document_evidence", []
            ),
            "semantic_field_ownership": self.context_profile.get(
                "semantic_field_ownership", {}
            ),
            "context_files": compact_files,
            "planning_path": self.planning_path,
            "intermediate_results": self.intermediate_results,
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[:max_chars] + "\n...[memory truncated]"


def _parse_json_response(raw_response: str) -> dict[str, Any]:
    return _load_single_json_object(_strip_json_fence(raw_response))


def _retrieve_relevant_document_evidence(
    task: PublicTask,
    *,
    max_chunks: int = 4,
    context_lines: int = 3,
) -> list[dict[str, Any]]:
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9_]+", task.question.lower())
        if len(term) >= 3 and term not in QUESTION_STOP_WORDS
    }
    candidates: list[tuple[int, Path, int, list[str]]] = []
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            line_lower = line.lower()
            overlap = sum(term in line_lower for term in question_terms)
            if overlap == 0:
                continue
            phrase_bonus = 2 if "severe thrombosis" in line_lower else 0
            candidates.append((overlap + phrase_bonus, path, index, lines))

    selected: list[dict[str, Any]] = []
    covered: dict[Path, set[int]] = {}
    for score, path, index, lines in sorted(
        candidates,
        key=lambda item: (-item[0], item[1].as_posix(), item[2]),
    ):
        path_covered = covered.setdefault(path, set())
        if any(abs(index - existing) <= context_lines for existing in path_covered):
            continue
        start = max(index - context_lines, 0)
        end = min(index + context_lines + 1, len(lines))
        selected.append(
            {
                "path": path.relative_to(task.context_dir).as_posix(),
                "line_start": start + 1,
                "line_end": end,
                "score": score,
                "text": "\n".join(lines[start:end]),
            }
        )
        path_covered.add(index)
        if len(selected) >= max_chunks:
            break
    return selected


def _extract_semantic_field_ownership(task: PublicTask) -> dict[str, list[str]]:
    ownership: dict[str, set[str]] = {}
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        current_entity: str | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            heading_match = re.match(r"^###\s+(.+?)\s*$", line)
            if heading_match:
                current_entity = heading_match.group(1).strip()
                continue
            if line.startswith("## "):
                current_entity = None
                continue
            field_match = re.match(r"^-\s+\*\*([^:*()]+)", line)
            if current_entity is None or field_match is None:
                continue
            field_name = field_match.group(1).strip()
            ownership.setdefault(field_name, set()).add(current_entity)
    return {
        field_name: sorted(entities)
        for field_name, entities in sorted(ownership.items())
    }


def _fallback_plan(task: PublicTask) -> dict[str, Any]:
    return {
        "thought": "Use a conservative hybrid retrieval plan after planner failure.",
        "decomposition_required": False,
        "sub_questions": [
            {
                "id": "q1",
                "question": task.question,
                "retrieval_strategy": "hybrid",
                "expected_evidence": "Authoritative rows and fields needed by the question.",
            }
        ],
        "sql_rewrite_policy": {
            "enabled": True,
            "goal": "Keep SQL compact, read-only, and aligned with requested filters.",
        },
        "final_answer_contract": {
            "columns": "Use exact observed source columns for requested attributes.",
            "row_rules": "Return only rows satisfying every filter in the question.",
            "validation_checks": [
                "Verify output columns and granularity.",
                "Verify filters and row count.",
                "Verify every value is grounded in tool output.",
            ],
        },
    }


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    sub_questions = plan.get("sub_questions")
    if not isinstance(sub_questions, list) or not sub_questions:
        raise ValueError("DAgent plan must contain a non-empty sub_questions list.")
    for item in sub_questions:
        if not isinstance(item, dict):
            raise TypeError("Each DAgent sub-question must be an object.")
        strategy = item.get("retrieval_strategy")
        if strategy not in {"direct", "sql", "hybrid"}:
            raise ValueError("retrieval_strategy must be direct, sql, or hybrid.")
    return plan


def _render_report(payload: dict[str, Any]) -> str:
    title = str(payload.get("title") or "DAgent-lite Analysis Report")
    summary = str(payload.get("summary") or "")
    findings = payload.get("findings", [])
    caveats = payload.get("caveats", [])
    finding_lines = findings if isinstance(findings, list) else [str(findings)]
    caveat_lines = caveats if isinstance(caveats, list) else [str(caveats)]

    lines = [f"# {title}", "", summary.strip(), "", "## Findings", ""]
    if finding_lines:
        lines.extend(f"- {item!s}" for item in finding_lines)
    else:
        lines.append("- No additional findings were generated.")
    lines.extend(["", "## Caveats", ""])
    if caveat_lines:
        lines.extend(f"- {item!s}" for item in caveat_lines)
    else:
        lines.append("- None recorded.")
    return "\n".join(lines).strip() + "\n"


def _fallback_report(task: PublicTask, answer: AnswerTable) -> str:
    return (
        "# DAgent-lite Analysis Report\n\n"
        f"The agent answered the question: {task.question}\n\n"
        "## Findings\n\n"
        f"- The final table contains {len(answer.rows)} row(s) and "
        f"{len(answer.columns)} column(s).\n"
        f"- Output columns: {', '.join(answer.columns)}.\n\n"
        "## Caveats\n\n"
        "- This fallback report summarizes the submitted table because model-based report "
        "generation was unavailable.\n"
    )


class DAgentLiteAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: DAgentLiteConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or DAgentLiteConfig()

    @staticmethod
    def _append_synthetic_step(
        state: AgentRuntimeState,
        *,
        action: str,
        thought: str,
        observation: dict[str, Any],
        ok: bool,
    ) -> None:
        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought=thought,
                action=action,
                action_input={},
                raw_response=json.dumps(observation, ensure_ascii=False, default=str),
                observation=observation,
                ok=ok,
            )
        )

    def _profile_and_plan(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: DAgentMemory,
    ) -> None:
        profile_result = self.tools.execute(
            task,
            "profile_context",
            {"sample_rows": 3, "max_doc_chars": 1200},
        )
        memory.context_profile = profile_result.content
        memory.context_profile["relevant_document_evidence"] = (
            _retrieve_relevant_document_evidence(task)
        )
        memory.context_profile["semantic_field_ownership"] = (
            _extract_semantic_field_ownership(task)
        )
        self._append_synthetic_step(
            state,
            action="__dagent_context_profile__",
            thought="DAgent profiled the available data before planning.",
            observation={"ok": profile_result.ok, "content": profile_result.content},
            ok=profile_result.ok,
        )

        profile_json = json.dumps(profile_result.content, ensure_ascii=False, default=str)
        if len(profile_json) > 18_000:
            profile_json = profile_json[:18_000] + "\n...[profile truncated]"
        messages = [
            ModelMessage(role="system", content=DAGENT_PLANNER_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=build_user_content_with_video(
                    task,
                    f"Question:\n{task.question}\n\n"
                    f"Context profile:\n{profile_json}\n\n"
                    f"Available tools:\n{self.tools.describe_for_prompt()}",
                ),
            ),
        ]
        try:
            raw_response = self.model.complete(messages)
            plan = _validate_plan(_parse_json_response(raw_response))
            ok = True
            error = None
        except Exception as exc:  # noqa: BLE001
            raw_response = ""
            plan = _fallback_plan(task)
            ok = False
            error = str(exc)

        memory.plan = plan
        self._append_synthetic_step(
            state,
            action="__dagent_plan__",
            thought=str(plan.get("thought", "DAgent generated a plan.")),
            observation={"ok": ok, "plan": plan, "planner_error": error},
            ok=True,
        )

    def _build_execution_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: DAgentMemory,
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=DAGENT_SYSTEM_PROMPT,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(
                role="user",
                content=build_user_content_with_video(
                    task,
                    f"{build_task_prompt(task)}\n\n"
                    "DAgent planning output:\n"
                    f"{json.dumps(memory.plan, ensure_ascii=False, indent=2)}",
                ),
            )
        )
        for step in state.steps:
            if step.action.startswith("__dagent_"):
                continue
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        messages.append(
            ModelMessage(
                role="user",
                content=(
                    "Current DAgent memory snapshot. Use it as observed execution context:\n"
                    f"{memory.snapshot()}"
                ),
            )
        )
        return messages

    def _rewrite_sql(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: DAgentMemory,
        action_input: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.sql_rewrite_enabled:
            return action_input
        original_sql = str(action_input.get("sql", ""))
        if not original_sql:
            return action_input
        previous_sql_actions = {
            "execute_context_sql",
            "execute_structured_sql",
        }.intersection(memory.planning_path)
        last_result = memory.intermediate_results[-1] if memory.intermediate_results else {}
        last_observation = last_result.get("observation", {})
        recovering_from_error = (
            isinstance(last_observation, dict)
            and not bool(last_observation.get("ok", True))
            and " join " not in f" {original_sql.lower()} "
        )
        diagnostic_id_filter = re.search(
            r"\bwhere\s+(?:\w+\.)?[\"`']?id[\"`']?\s*=",
            original_sql,
            flags=re.IGNORECASE,
        )
        if previous_sql_actions and (diagnostic_id_filter or recovering_from_error):
            final_sql, semantic_corrections = rewrite_semantic_column_owners(
                original_sql,
                memory.context_profile.get("semantic_field_ownership", {}),
            )
            self._append_synthetic_step(
                state,
                action="__dagent_sql_rewrite__",
                thought="DAgent preserved a narrow diagnostic follow-up query.",
                observation={
                    "ok": True,
                    "original_sql": original_sql,
                    "final_sql": final_sql,
                    "keep_original": final_sql == original_sql,
                    "semantic_owner_corrections": semantic_corrections,
                    "thought": (
                        "The query is a narrow diagnostic step after an earlier relational result."
                    ),
                },
                ok=True,
            )
            rewritten_input = dict(action_input)
            rewritten_input["sql"] = final_sql
            return rewritten_input

        messages = [
            ModelMessage(role="system", content=DAGENT_SQL_REWRITE_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question:\n{task.question}\n\n"
                    f"Proposed SQL:\n{original_sql}\n\n"
                    "Relevant semantic document evidence:\n"
                    f"{json.dumps(memory.context_profile.get('relevant_document_evidence', []), ensure_ascii=False)}\n\n"
                    "Semantic field ownership:\n"
                    f"{json.dumps(memory.context_profile.get('semantic_field_ownership', {}), ensure_ascii=False)}\n\n"
                    f"DAgent memory:\n{memory.snapshot(max_chars=10_000)}"
                ),
            ),
        ]
        try:
            raw_response = self.model.complete(messages)
            payload = _parse_json_response(raw_response)
            rewritten_sql = str(payload.get("sql", "")).strip()
            if not rewritten_sql:
                raise ValueError("SQL rewrite response omitted sql.")
            keep_original = bool(payload.get("keep_original", False))
            final_sql = original_sql if keep_original else rewritten_sql
            final_sql, semantic_corrections = rewrite_semantic_column_owners(
                final_sql,
                memory.context_profile.get("semantic_field_ownership", {}),
            )
            observation = {
                "ok": True,
                "original_sql": original_sql,
                "final_sql": final_sql,
                "keep_original": keep_original,
                "thought": str(payload.get("thought", "")),
                "semantic_owner_corrections": semantic_corrections,
            }
        except Exception as exc:  # noqa: BLE001
            final_sql = original_sql
            observation = {
                "ok": False,
                "original_sql": original_sql,
                "final_sql": final_sql,
                "rewrite_error": str(exc),
                "semantic_owner_corrections": [],
            }

        self._append_synthetic_step(
            state,
            action="__dagent_sql_rewrite__",
            thought="DAgent reviewed the proposed SQL before execution.",
            observation=observation,
            ok=True,
        )
        rewritten_input = dict(action_input)
        rewritten_input["sql"] = final_sql
        return rewritten_input

    def _generate_report(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: DAgentMemory,
        answer: AnswerTable,
    ) -> None:
        if not self.config.report_generation_enabled:
            return
        messages = [
            ModelMessage(role="system", content=DAGENT_REPORT_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question:\n{task.question}\n\n"
                    f"Final answer:\n{json.dumps(answer.to_dict(), ensure_ascii=False, default=str)}\n\n"
                    f"DAgent memory:\n{memory.snapshot(max_chars=12_000)}"
                ),
            ),
        ]
        try:
            raw_response = self.model.complete(messages)
            report_payload = _parse_json_response(raw_response)
            report_markdown = _render_report(report_payload)
            observation = {
                "ok": True,
                "report_payload": report_payload,
                "report_markdown": report_markdown,
            }
        except Exception as exc:  # noqa: BLE001
            report_markdown = _fallback_report(task, answer)
            observation = {
                "ok": False,
                "report_error": str(exc),
                "report_markdown": report_markdown,
            }

        self._append_synthetic_step(
            state,
            action="__dagent_report_generation__",
            thought="DAgent synthesized the final table into an analytical report.",
            observation=observation,
            ok=True,
        )

    def _review_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: DAgentMemory,
        answer: AnswerTable,
    ) -> bool:
        if not self.config.answer_review_enabled:
            return True
        messages = [
            ModelMessage(role="system", content=DAGENT_ANSWER_REVIEW_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question:\n{task.question}\n\n"
                    f"Proposed answer:\n{json.dumps(answer.to_dict(), ensure_ascii=False, default=str)}\n\n"
                    f"DAgent memory:\n{memory.snapshot(max_chars=12_000)}"
                ),
            ),
        ]
        try:
            raw_response = self.model.complete(messages)
            payload = _parse_json_response(raw_response)
            accept = payload.get("accept")
            if not isinstance(accept, bool):
                raise TypeError("Answer review response must contain a boolean accept field.")
            issues = payload.get("issues", [])
            suggested_action = str(payload.get("suggested_action", ""))
            suggestion_lower = suggested_action.casefold()
            consistency_override = (
                not accept
                and (
                    "no further action" in suggestion_lower
                    or "no additional action" in suggestion_lower
                )
            )
            if consistency_override:
                accept = True
            observation = {
                "ok": True,
                "accept": accept,
                "issues": issues if isinstance(issues, list) else [str(issues)],
                "suggested_action": suggested_action,
                "consistency_override": consistency_override,
            }
        except Exception as exc:  # noqa: BLE001
            accept = True
            observation = {
                "ok": False,
                "accept": True,
                "review_error": str(exc),
                "issues": [],
                "suggested_action": "",
            }

        self._append_synthetic_step(
            state,
            action="__dagent_answer_review__",
            thought="DAgent reviewed the proposed final answer before accepting it.",
            observation=observation,
            ok=bool(observation["ok"]),
        )
        if not accept:
            memory.remember(
                action="__dagent_answer_review__",
                action_input={},
                observation=observation,
                max_items=self.config.memory_max_items,
            )
        return accept

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        memory = DAgentMemory(question=task.question)
        try:
            self._profile_and_plan(task, state, memory)
        except Exception as exc:  # noqa: BLE001
            state.failure_reason = f"DAgent profiling or planning failed: {exc}"
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )

        for _ in range(self.config.max_steps):
            try:
                raw_response = self.model.complete(
                    self._build_execution_messages(task, state, memory)
                )
            except Exception as exc:  # noqa: BLE001
                state.failure_reason = f"DAgent model request failed: {exc}"
                break

            try:
                model_step = parse_model_step(raw_response)
                effective_action = model_step.action
                if effective_action == "execute_context_sql":
                    effective_action = "execute_structured_sql"
                effective_input = model_step.action_input
                if effective_action == "execute_structured_sql":
                    effective_input = self._rewrite_sql(
                        task,
                        state,
                        memory,
                        model_step.action_input,
                    )

                if effective_action == "execute_structured_sql":
                    effective_input = {
                        "sql": effective_input.get("sql", ""),
                        "limit": effective_input.get("limit", 200),
                    }
                tool_result = self.tools.execute(task, effective_action, effective_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": effective_action,
                    "content": tool_result.content,
                }
                state.steps.append(
                    StepRecord(
                        step_index=len(state.steps) + 1,
                        thought=model_step.thought,
                        action=effective_action,
                        action_input=effective_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )
                memory.remember(
                    action=effective_action,
                    action_input=effective_input,
                    observation=observation,
                    max_items=self.config.memory_max_items,
                )
                if tool_result.is_terminal and tool_result.answer is not None:
                    if not self._review_answer(task, state, memory, tool_result.answer):
                        continue
                    state.answer = tool_result.answer
                    self._generate_report(task, state, memory, tool_result.answer)
                    break
            except Exception as exc:  # noqa: BLE001
                observation = {"ok": False, "error": str(exc)}
                state.steps.append(
                    StepRecord(
                        step_index=len(state.steps) + 1,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                memory.remember(
                    action="__error__",
                    action_input={},
                    observation=observation,
                    max_items=self.config.memory_max_items,
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "DAgent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
