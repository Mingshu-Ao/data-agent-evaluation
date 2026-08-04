from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.agentic_memory import (
    AgenticLongTermMemory,
    bellman_state_values,
)
from data_agent_baseline.agents.agentic_optimizer import optimize_semantic_plan
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.multimodal import build_user_content_with_video
from data_agent_baseline.agents.prompt import build_observation_prompt, build_task_prompt
from data_agent_baseline.agents.react import (
    _load_single_json_object,
    _strip_json_fence,
    parse_model_step,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

AGENTIC_DATA_PLANNER_PROMPT = """
You are the query planning agent of AgenticData-lite, a paper-inspired multi-agent analytics system.

Translate the natural-language question into a semantic logical plan grounded in the supplied profile
graph. Use relational operators for structured data and semantic operators only when text meaning
cannot be resolved with relational operations.

Allowed operators:
Scan, Filter, Project, Join, Aggregate, Union, Intersect, Sort, Limit,
SemanticScan, SemanticFilter, SemanticExtract, SemanticJoin, Validate, Generate.

Return exactly one fenced ```json block containing one object with:
- thought: concise rationale
- task_decomposition: non-empty list of objects with id, question, and dependencies
- selected_data: non-empty list of objects with node_id and reason
- logical_plan: non-empty ordered list of objects with id, operator, inputs, source_nodes,
  instruction, and output_columns
- answer_contract: object with columns, granularity, and checks

Every dependency and plan input must refer to an earlier id. Every source_nodes item must be a node
from the profile graph. Do not solve the task and do not output text outside the JSON block.
""".strip()

AGENTIC_DATA_SEMANTIC_VALIDATOR_PROMPT = """
You are the semantic plan validator of AgenticData-lite.

Check whether the proposed logical plan uses the right datasets, covers every condition and requested
attribute, preserves the requested answer granularity, and contains enough evidence-producing steps.
Do not reject a plan solely because aliases differ from benchmark-specific SQL expression headers.

Return exactly one fenced ```json block containing one object with:
- accept: boolean
- issues: list of concise strings
- transition_feedback: a concise instruction for the next planning attempt, or an empty string

Do not output text outside the JSON block.
""".strip()

AGENTIC_DATA_EXECUTOR_PROMPT = """
You are the execution agent of AgenticData-lite.

Follow the validated semantic logical plan and ground all values in tool observations.

Rules:
1. Use execute_structured_sql for SQLite, JSON, CSV, and Parquet relational work, including
   cross-source joins. Use table names, not file paths, in SQL FROM and JOIN clauses.
2. Use document tools or Python only when relational operations cannot answer the question.
3. Respect the answer contract's columns and granularity.
4. Never invent data or silently replace missing values.
5. Finish only by calling answer with columns and rows.
6. Return exactly one fenced ```json block with thought, action, and action_input.
""".strip()

AGENTIC_DATA_ANSWER_VALIDATOR_PROMPT = """
You are the final answer validator of AgenticData-lite.

Check the proposed table against the question, validated semantic plan, answer contract, and observed
evidence. Reject wrong fields, missing rows, unsupported values, filter violations, or incorrect
granularity.

Return exactly one fenced ```json block containing:
- accept: boolean
- issues: list of concise strings
- transition_feedback: concise next execution step if rejected, otherwise an empty string

Do not output text outside the JSON block.
""".strip()

ALLOWED_OPERATORS = {
    "Aggregate",
    "Filter",
    "Generate",
    "Intersect",
    "Join",
    "Limit",
    "Project",
    "Scan",
    "SemanticExtract",
    "SemanticFilter",
    "SemanticJoin",
    "SemanticScan",
    "Sort",
    "Union",
    "Validate",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class AgenticDataLiteConfig:
    max_steps: int = 16
    max_plan_revisions: int = 1
    memory_max_items: int = 12
    answer_validation_enabled: bool = True
    profile_enabled: bool = True
    edge_profile_enabled: bool = True
    plan_validation_enabled: bool = True
    memory_enabled: bool = True
    optimizer_enabled: bool = True
    long_term_memory_path: Path | None = None


@dataclass(slots=True)
class AgenticDataMemory:
    question: str
    profile_graph: dict[str, Any] = field(default_factory=dict)
    logical_plan: dict[str, Any] = field(default_factory=dict)
    optimized_plan: dict[str, Any] = field(default_factory=dict)
    transition_feedback: list[str] = field(default_factory=list)
    plan_transitions: list[dict[str, Any]] = field(default_factory=list)
    state_values: list[float] = field(default_factory=list)
    retrieved_long_term_memories: list[dict[str, Any]] = field(default_factory=list)
    execution_history: list[dict[str, Any]] = field(default_factory=list)

    def add_feedback(self, feedback: str) -> None:
        normalized = feedback.strip()
        if normalized:
            self.transition_feedback.append(normalized)

    def remember_execution(
        self,
        *,
        action: str,
        action_input: dict[str, Any],
        observation: dict[str, Any],
        max_items: int,
    ) -> None:
        self.execution_history.append(
            {
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
        )
        if len(self.execution_history) > max_items:
            self.execution_history = self.execution_history[-max_items:]

    def remember_plan_transition(
        self,
        *,
        plan: dict[str, Any],
        accepted: bool,
        feedback: str,
    ) -> None:
        self.plan_transitions.append(
            {
                "plan": plan,
                "accepted": accepted,
                "feedback": feedback,
                "reward": 1.0 if accepted else -1.0,
            }
        )
        self.state_values = bellman_state_values(self.plan_transitions)

    def snapshot(self, *, max_chars: int = 14_000) -> str:
        payload = {
            "question": self.question,
            "logical_plan": self.logical_plan,
            "optimized_plan": self.optimized_plan,
            "transition_feedback": self.transition_feedback[-6:],
            "state_values": self.state_values,
            "retrieved_long_term_memories": self.retrieved_long_term_memories,
            "execution_history": self.execution_history,
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[:max_chars] + "\n...[memory truncated]"


def _parse_json_response(raw_response: str) -> dict[str, Any]:
    return _load_single_json_object(_strip_json_fence(raw_response))


def _tokenize(*values: Any) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        for token in TOKEN_PATTERN.findall(str(value).lower()):
            if len(token) >= 2:
                tokens.add(token)
    return sorted(tokens)


def _json_columns(summary: Any) -> list[str]:
    if not isinstance(summary, dict):
        return []
    sample = summary.get("sample")
    if not isinstance(sample, dict):
        return list(summary.get("keys", [])) if isinstance(summary.get("keys"), list) else []
    records = sample.get("records")
    if isinstance(records, dict):
        item_sample = records.get("item_sample")
        if isinstance(item_sample, dict) and isinstance(item_sample.get("keys"), list):
            return [str(item) for item in item_sample["keys"]]
    return [str(item) for item in summary.get("keys", [])] if isinstance(summary.get("keys"), list) else []


def build_profile_graph(profile: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for item in profile.get("files", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "unknown"))
        kind = str(item.get("kind", "file"))
        if kind == "sqlite":
            for table in item.get("tables", []):
                if not isinstance(table, dict):
                    continue
                columns = [str(value) for value in table.get("columns", [])]
                table_name = str(table.get("name", "unknown"))
                nodes.append(
                    {
                        "node_id": f"{path}::{table_name}",
                        "path": path,
                        "kind": "structured_table",
                        "table_name": table_name,
                        "columns": columns,
                        "row_count": table.get("row_count"),
                        "summary": f"SQLite table {table_name} with columns {columns}.",
                        "keywords": _tokenize(path, table_name, *columns),
                    }
                )
            continue

        if kind == "csv":
            columns = [str(value) for value in item.get("columns", [])]
            table_name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            nodes.append(
                {
                    "node_id": path,
                    "path": path,
                    "kind": "structured_table",
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": item.get("row_count"),
                    "summary": f"CSV table {table_name} with columns {columns}.",
                    "keywords": _tokenize(path, table_name, *columns),
                }
            )
            continue

        if kind == "json":
            columns = _json_columns(item.get("summary", {}))
            nodes.append(
                {
                    "node_id": path,
                    "path": path,
                    "kind": "semi_structured",
                    "table_name": path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
                    "columns": columns,
                    "summary": f"JSON source with inferred fields {columns}.",
                    "keywords": _tokenize(path, *columns),
                }
            )
            continue

        if kind == "document":
            preview = str(item.get("preview", ""))
            nodes.append(
                {
                    "node_id": path,
                    "path": path,
                    "kind": "unstructured_document",
                    "columns": [],
                    "char_count": item.get("char_count"),
                    "summary": preview[:600],
                    "keywords": _tokenize(path, preview[:1200]),
                }
            )
            continue

        nodes.append(
            {
                "node_id": path,
                "path": path,
                "kind": kind,
                "columns": [],
                "summary": f"Context file {path}.",
                "keywords": _tokenize(path),
            }
        )

    edges: list[dict[str, Any]] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            left_columns = {str(value).lower() for value in left.get("columns", [])}
            right_columns = {str(value).lower() for value in right.get("columns", [])}
            shared_columns = sorted(left_columns & right_columns)
            left_keywords = set(left.get("keywords", []))
            right_keywords = set(right.get("keywords", []))
            shared_keywords = sorted(left_keywords & right_keywords)
            union = left_keywords | right_keywords
            keyword_similarity = len(shared_keywords) / len(union) if union else 0.0
            if not shared_columns and keyword_similarity < 0.05:
                continue
            edges.append(
                {
                    "source": left["node_id"],
                    "target": right["node_id"],
                    "relationship": "potential_join_or_semantic_link",
                    "shared_columns": shared_columns,
                    "shared_keywords": shared_keywords[:20],
                    "score": round(len(shared_columns) + keyword_similarity, 4),
                }
            )

    edges.sort(key=lambda item: (-float(item["score"]), item["source"], item["target"]))
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges[: max(len(nodes) * 10, 20)],
    }


def _fallback_plan(task: PublicTask, graph: dict[str, Any]) -> dict[str, Any]:
    node_ids = [str(node["node_id"]) for node in graph.get("nodes", [])]
    return {
        "thought": "Use a conservative profile-grounded plan after planner validation failed.",
        "task_decomposition": [
            {
                "id": "task_1",
                "question": task.question,
                "dependencies": [],
            }
        ],
        "selected_data": [
            {"node_id": node_id, "reason": "Candidate source from the profile graph."}
            for node_id in node_ids
        ],
        "logical_plan": [
            {
                "id": "op_1",
                "operator": "Scan",
                "inputs": [],
                "source_nodes": node_ids,
                "instruction": "Inspect relevant structured and document sources.",
                "output_columns": [],
            },
            {
                "id": "op_2",
                "operator": "Validate",
                "inputs": ["op_1"],
                "source_nodes": node_ids,
                "instruction": "Validate filters, requested columns, and answer granularity.",
                "output_columns": [],
            },
            {
                "id": "op_3",
                "operator": "Generate",
                "inputs": ["op_2"],
                "source_nodes": [],
                "instruction": "Submit the grounded answer table.",
                "output_columns": [],
            },
        ],
        "answer_contract": {
            "columns": "Use requested source attributes.",
            "granularity": "One row per entity requested by the question.",
            "checks": [
                "All filters are applied.",
                "All answer values are grounded in tool observations.",
            ],
        },
    }


def validate_plan_grammar(plan: dict[str, Any], graph: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    graph_nodes = {str(node.get("node_id")) for node in graph.get("nodes", [])}
    selected_data = plan.get("selected_data")
    if not isinstance(selected_data, list) or not selected_data:
        issues.append("selected_data must be a non-empty list.")
    else:
        for item in selected_data:
            node_id = item.get("node_id") if isinstance(item, dict) else None
            if node_id not in graph_nodes:
                issues.append(f"Unknown selected data node: {node_id}.")

    decomposition = plan.get("task_decomposition")
    if not isinstance(decomposition, list) or not decomposition:
        issues.append("task_decomposition must be a non-empty list.")

    logical_plan = plan.get("logical_plan")
    if not isinstance(logical_plan, list) or not logical_plan:
        issues.append("logical_plan must be a non-empty list.")
        return issues

    seen_ids: set[str] = set()
    for step in logical_plan:
        if not isinstance(step, dict):
            issues.append("Every logical plan step must be an object.")
            continue
        step_id = str(step.get("id", "")).strip()
        operator = str(step.get("operator", "")).strip()
        if not step_id:
            issues.append("Every logical plan step must have an id.")
        elif step_id in seen_ids:
            issues.append(f"Duplicate logical plan id: {step_id}.")
        for dependency in step.get("inputs", []):
            if dependency not in seen_ids:
                issues.append(
                    f"Plan step {step_id or '<missing>'} references non-prior input {dependency}."
                )
        for node_id in step.get("source_nodes", []):
            if node_id not in graph_nodes:
                issues.append(f"Plan step {step_id or '<missing>'} uses unknown node {node_id}.")
        if operator not in ALLOWED_OPERATORS:
            issues.append(f"Unsupported operator: {operator or '<missing>'}.")
        if step_id:
            seen_ids.add(step_id)

    return issues


class AgenticDataLiteAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: AgenticDataLiteConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or AgenticDataLiteConfig()
        self.long_term_memory = AgenticLongTermMemory(
            self.config.long_term_memory_path
        )

    @staticmethod
    def _append_step(
        state: AgentRuntimeState,
        *,
        action: str,
        thought: str,
        observation: dict[str, Any],
        ok: bool,
        raw_response: str = "",
    ) -> None:
        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought=thought,
                action=action,
                action_input={},
                raw_response=raw_response or json.dumps(observation, ensure_ascii=False, default=str),
                observation=observation,
                ok=ok,
            )
        )

    def _build_profile_graph(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: AgenticDataMemory,
    ) -> None:
        profile_result = self.tools.execute(
            task,
            "profile_context",
            {"sample_rows": 3, "max_doc_chars": 1800},
        )
        graph = build_profile_graph(profile_result.content)
        if not self.config.profile_enabled:
            graph["nodes"] = [
                {
                    "node_id": node.get("node_id"),
                    "path": node.get("path"),
                    "kind": node.get("kind"),
                    "columns": [],
                    "summary": "",
                    "keywords": [],
                }
                for node in graph.get("nodes", [])
            ]
            graph["edges"] = []
            graph["edge_count"] = 0
        elif not self.config.edge_profile_enabled:
            graph["edges"] = []
            graph["edge_count"] = 0
        memory.profile_graph = graph
        self._append_step(
            state,
            action="__agenticdata_profile_graph__",
            thought="AgenticData-lite profiled heterogeneous sources and linked related nodes.",
            observation={
                "ok": profile_result.ok,
                "profile_graph": graph,
                "profile_errors": profile_result.content.get("errors", []),
            },
            ok=profile_result.ok,
        )

    def _planner_messages(
        self,
        task: PublicTask,
        memory: AgenticDataMemory,
    ) -> list[ModelMessage]:
        graph_json = json.dumps(memory.profile_graph, ensure_ascii=False, indent=2, default=str)
        if len(graph_json) > 20_000:
            graph_json = graph_json[:20_000] + "\n...[profile graph truncated]"
        feedback = "\n".join(f"- {item}" for item in memory.transition_feedback[-6:]) or "- None"
        retrieved = (
            json.dumps(
                memory.retrieved_long_term_memories,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            if memory.retrieved_long_term_memories
            else "[]"
        )
        return [
            ModelMessage(role="system", content=AGENTIC_DATA_PLANNER_PROMPT),
            ModelMessage(
                role="user",
                content=build_user_content_with_video(
                    task,
                    f"Question:\n{task.question}\n\n"
                    f"Profile graph:\n{graph_json}\n\n"
                    f"Transition feedback from prior attempts:\n{feedback}\n\n"
                    f"Relevant long-term plan memories:\n{retrieved}",
                ),
            ),
        ]

    def _semantic_validator_messages(
        self,
        task: PublicTask,
        memory: AgenticDataMemory,
        plan: dict[str, Any],
    ) -> list[ModelMessage]:
        return [
            ModelMessage(role="system", content=AGENTIC_DATA_SEMANTIC_VALIDATOR_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question:\n{task.question}\n\n"
                    f"Profile graph:\n"
                    f"{json.dumps(memory.profile_graph, ensure_ascii=False, indent=2)}\n\n"
                    f"Proposed plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
                ),
            ),
        ]

    def _plan(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: AgenticDataMemory,
    ) -> None:
        accepted_plan: dict[str, Any] | None = None
        attempts = self.config.max_plan_revisions + 1
        for attempt in range(attempts):
            try:
                raw_plan = self.model.complete(self._planner_messages(task, memory))
                plan = _parse_json_response(raw_plan)
            except Exception as exc:  # noqa: BLE001
                memory.add_feedback(f"Planner response error: {exc}")
                self._append_step(
                    state,
                    action="__agenticdata_plan__",
                    thought=f"Planning attempt {attempt + 1} could not be parsed.",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                )
                continue

            self._append_step(
                state,
                action="__agenticdata_plan__",
                thought=str(plan.get("thought", f"Planning attempt {attempt + 1}.")),
                observation={"ok": True, "attempt": attempt + 1, "plan": plan},
                ok=True,
                raw_response=raw_plan,
            )

            grammar_issues = validate_plan_grammar(plan, memory.profile_graph)
            self._append_step(
                state,
                action="__agenticdata_grammar_validation__",
                thought="Validate plan structure, operators, dependencies, and source references.",
                observation={
                    "ok": not grammar_issues,
                    "issues": grammar_issues,
                    "attempt": attempt + 1,
                },
                ok=not grammar_issues,
            )
            if grammar_issues:
                memory.add_feedback("Grammar validation: " + " ".join(grammar_issues))
                memory.remember_plan_transition(
                    plan=plan,
                    accepted=False,
                    feedback=" ".join(grammar_issues),
                )
                continue

            if not self.config.plan_validation_enabled:
                memory.remember_plan_transition(
                    plan=plan,
                    accepted=True,
                    feedback="Plan validation disabled by ablation.",
                )
                accepted_plan = plan
                break

            try:
                raw_validation = self.model.complete(
                    self._semantic_validator_messages(task, memory, plan)
                )
                validation = _parse_json_response(raw_validation)
                accepted = bool(validation.get("accept"))
                issues = validation.get("issues", [])
                feedback = str(validation.get("transition_feedback", ""))
            except Exception as exc:  # noqa: BLE001
                raw_validation = ""
                accepted = False
                issues = [f"Semantic validator response error: {exc}"]
                feedback = str(exc)

            self._append_step(
                state,
                action="__agenticdata_semantic_validation__",
                thought="Validate semantic completeness before execution.",
                observation={
                    "ok": accepted,
                    "accept": accepted,
                    "issues": issues,
                    "transition_feedback": feedback,
                    "attempt": attempt + 1,
                },
                ok=accepted,
                raw_response=raw_validation,
            )
            if accepted:
                memory.remember_plan_transition(
                    plan=plan,
                    accepted=True,
                    feedback=feedback,
                )
                accepted_plan = plan
                break
            memory.remember_plan_transition(
                plan=plan,
                accepted=False,
                feedback=feedback or "Semantic validation rejected the proposed plan.",
            )
            memory.add_feedback(feedback or "Semantic validation rejected the proposed plan.")

        memory.logical_plan = accepted_plan or _fallback_plan(task, memory.profile_graph)
        if accepted_plan is None:
            self._append_step(
                state,
                action="__agenticdata_fallback_plan__",
                thought="Use a conservative profile-grounded fallback after planning retries.",
                observation={"ok": True, "plan": memory.logical_plan},
                ok=True,
            )

    def _optimize_plan(
        self,
        state: AgentRuntimeState,
        memory: AgenticDataMemory,
    ) -> None:
        if not self.config.optimizer_enabled:
            memory.optimized_plan = memory.logical_plan
            return
        memory.optimized_plan = optimize_semantic_plan(
            memory.logical_plan,
            memory.profile_graph,
        )
        memory.logical_plan = memory.optimized_plan
        self._append_step(
            state,
            action="__agenticdata_plan_optimization__",
            thought="Estimate operator costs and attach a physical execution strategy.",
            observation={
                "ok": True,
                "optimizer": memory.optimized_plan.get("optimizer", {}),
                "physical_plan": memory.optimized_plan.get("physical_plan", []),
            },
            ok=True,
        )

    def _persist_plan_memory(
        self,
        task: PublicTask,
        memory: AgenticDataMemory,
    ) -> None:
        if not self.config.memory_enabled:
            return
        for index, transition in enumerate(memory.plan_transitions):
            self.long_term_memory.store(
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "kind": "good_plan" if transition["accepted"] else "bad_plan",
                    "plan": transition["plan"],
                    "feedback": transition["feedback"],
                    "reward": transition["reward"],
                    "state_value": memory.state_values[index],
                }
            )

    def _execution_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        memory: AgenticDataMemory,
    ) -> list[ModelMessage]:
        messages = [
            ModelMessage(
                role="system",
                content=(
                    f"{AGENTIC_DATA_EXECUTOR_PROMPT}\n\n"
                    f"Available tools:\n{self.tools.describe_for_prompt()}"
                ),
            ),
            ModelMessage(
                role="user",
                content=build_user_content_with_video(
                    task,
                    f"{build_task_prompt(task)}\n\n"
                    "Validated AgenticData semantic plan:\n"
                    f"{json.dumps(memory.logical_plan, ensure_ascii=False, indent=2)}",
                ),
            ),
        ]
        for step in state.steps:
            if step.action.startswith("__agenticdata_"):
                continue
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        messages.append(
            ModelMessage(
                role="user",
                content=f"Current feedback-aware memory:\n{memory.snapshot()}",
            )
        )
        return messages

    def _answer_validation_messages(
        self,
        task: PublicTask,
        memory: AgenticDataMemory,
        answer_input: dict[str, Any],
    ) -> list[ModelMessage]:
        return [
            ModelMessage(role="system", content=AGENTIC_DATA_ANSWER_VALIDATOR_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question:\n{task.question}\n\n"
                    f"Validated plan:\n"
                    f"{json.dumps(memory.logical_plan, ensure_ascii=False, indent=2)}\n\n"
                    f"Proposed answer:\n{json.dumps(answer_input, ensure_ascii=False, indent=2)}\n\n"
                    f"Execution memory:\n{memory.snapshot()}"
                ),
            ),
        ]

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        memory = AgenticDataMemory(question=task.question)
        try:
            self._build_profile_graph(task, state, memory)
            if self.config.memory_enabled:
                memory.retrieved_long_term_memories = self.long_term_memory.retrieve(
                    task.question
                )
                self._append_step(
                    state,
                    action="__agenticdata_memory_retrieval__",
                    thought="Retrieve related good and bad plans from long-term memory.",
                    observation={
                        "ok": True,
                        "retrieved_count": len(
                            memory.retrieved_long_term_memories
                        ),
                        "records": [
                            {
                                "task_id": item.get("task_id"),
                                "kind": item.get("kind"),
                                "state_value": item.get("state_value"),
                                "retrieval_score": item.get("retrieval_score"),
                            }
                            for item in memory.retrieved_long_term_memories
                        ],
                    },
                    ok=True,
                )
            self._plan(task, state, memory)
            self._optimize_plan(state, memory)
            self._persist_plan_memory(task, memory)
        except Exception as exc:  # noqa: BLE001
            state.failure_reason = f"AgenticData-lite planning failed: {exc}"
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )

        for _ in range(self.config.max_steps):
            try:
                raw_response = self.model.complete(
                    self._execution_messages(task, state, memory)
                )
                model_step = parse_model_step(raw_response)
            except Exception as exc:  # noqa: BLE001
                state.failure_reason = f"AgenticData-lite model request failed: {exc}"
                break

            if model_step.action == "answer" and self.config.answer_validation_enabled:
                try:
                    raw_validation = self.model.complete(
                        self._answer_validation_messages(
                            task,
                            memory,
                            model_step.action_input,
                        )
                    )
                    validation = _parse_json_response(raw_validation)
                    accepted = bool(validation.get("accept"))
                    feedback = str(validation.get("transition_feedback", ""))
                    issues = validation.get("issues", [])
                    self._append_step(
                        state,
                        action="__agenticdata_answer_validation__",
                        thought="Validate the final table against the plan and evidence.",
                        observation={
                            "ok": accepted,
                            "accept": accepted,
                            "issues": issues,
                            "transition_feedback": feedback,
                        },
                        ok=accepted,
                        raw_response=raw_validation,
                    )
                    if not accepted:
                        memory.add_feedback(feedback or "Final answer validation failed.")
                        continue
                except Exception as exc:  # noqa: BLE001
                    self._append_step(
                        state,
                        action="__agenticdata_answer_validation__",
                        thought="Answer validation was unavailable; preserve the grounded answer.",
                        observation={"ok": False, "validator_error": str(exc)},
                        ok=False,
                    )

            try:
                tool_result = self.tools.execute(
                    task,
                    model_step.action,
                    model_step.action_input,
                )
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
            except Exception as exc:  # noqa: BLE001
                observation = {
                    "ok": False,
                    "tool": model_step.action,
                    "error": str(exc),
                }
                memory.add_feedback(f"Execution error for {model_step.action}: {exc}")
                tool_result = None

            state.steps.append(
                StepRecord(
                    step_index=len(state.steps) + 1,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=bool(observation["ok"]),
                )
            )
            if self.config.memory_enabled:
                memory.remember_execution(
                    action=model_step.action,
                    action_input=model_step.action_input,
                    observation=observation,
                    max_items=self.config.memory_max_items,
                )
            if tool_result is not None and tool_result.is_terminal:
                state.answer = tool_result.answer
                if self.config.memory_enabled:
                    self.long_term_memory.store(
                        {
                            "task_id": task.task_id,
                            "question": task.question,
                            "kind": "successful_context",
                            "plan": memory.logical_plan,
                            "feedback": "",
                            "reward": 1.0,
                            "state_value": 1.0,
                        }
                    )
                break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "AgenticData-lite did not submit an answer within max_steps."
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
