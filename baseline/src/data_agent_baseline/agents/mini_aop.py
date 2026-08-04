from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.multimodal import build_user_content_with_video
from data_agent_baseline.agents.operators import get_operator_spec, normalize_operator_name
from data_agent_baseline.agents.parallel_executor import execute_parallel_prefetch
from data_agent_baseline.agents.prompt import (
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react import (
    ReActAgentConfig,
    _load_single_json_object,
    _strip_json_fence,
    parse_model_step,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

MINI_AOP_SYSTEM_PROMPT = """
You are a Mini-AOP data agent inspired by automated LLM pipeline orchestration.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Follow the provided Mini-AOP operator-level pipeline plan unless observations prove it wrong.
2. Treat the plan as a contract: identify sources, link entities, apply filters, extract requested columns, and validate before answering.
3. Prefer executable Python or SQL over reasoning from truncated previews when a complete table/file must be analyzed.
4. For requested entity attributes, use the source that actually defines that entity. Do not fill missing requested attributes from unrelated records.
5. If a row cannot provide all requested output attributes from observed authoritative sources, omit it unless the task explicitly asks for incomplete rows.
6. Use exact source column names for final output columns when requested attributes map to observed fields, for example use `SEX` instead of `sex` and `Diagnosis` instead of `disease` if those are the source fields.
7. Preserve the requested output granularity: if a question asks for full names but the observed data has `first_name` and `last_name`, return those separate observed fields unless the question explicitly asks for one combined string.
8. Before calling `answer`, validate three things: output columns, row count/filter correctness, and whether any column was renamed or merged without evidence.
9. Base your answer only on information observed through tools.
10. The task is complete only when you call the `answer` tool.
11. The `answer` tool must receive a table with `columns` and `rows`.
12. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
13. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
14. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()


MINI_AOP_PLAN_SYSTEM_PROMPT = """
You are a Mini-AOP planner for data-agent tasks.

Given a natural-language question, the available context files, and the available tools, generate an operator-level execution pipeline inspired by AOP.

Use only these operator names when useful:
Plan, Link, Scan, Retrieve, Filter, Extract, Transform, GroupBy, Aggregate, Compare, Validate, Generate.

Return exactly one fenced ```json block containing one JSON object with keys:
- thought: short string
- pipeline: list of operator steps. Each step must include `op`, `goal`, `sources`, and `success_criteria`.
- final_answer_contract: object with `columns`, `row_rules`, and `validation_checks`

Do not solve the task. Plan how to solve it robustly.
When final output attributes correspond to observed source fields, plan to use the exact source column names in the final table.
When a requested value can be represented either as a merged natural-language phrase or as multiple observed fields, prefer the observed field-level columns unless the question explicitly requires a merged value.
Include a validation check that compares the planned final columns against the observed source fields before submitting the answer.
""".strip()


def _parse_plan(raw_response: str) -> dict[str, Any]:
    payload = _load_single_json_object(_strip_json_fence(raw_response))
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("Mini-AOP plan must include a non-empty pipeline list.")
    return payload


def _plan_sources(plan: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    for step in plan.get("pipeline", []):
        if not isinstance(step, dict):
            continue
        raw_sources = step.get("sources", [])
        if isinstance(raw_sources, str):
            sources.add(raw_sources)
        elif isinstance(raw_sources, list):
            sources.update(str(source) for source in raw_sources)
    return {source for source in sources if source and source.lower() not in {"n/a", "none", "unknown"}}


def _score_plan(plan: dict[str, Any]) -> dict[str, Any]:
    pipeline = [step for step in plan.get("pipeline", []) if isinstance(step, dict)]
    ops = [normalize_operator_name(str(step.get("op", ""))) for step in pipeline]
    contract = plan.get("final_answer_contract", {})
    validation_checks = []
    if isinstance(contract, dict):
        raw_checks = contract.get("validation_checks", [])
        if isinstance(raw_checks, list):
            validation_checks = raw_checks

    required_ops = {"plan", "scan", "filter", "extract", "validate", "generate"}
    optional_ops = {"link", "aggregate", "groupby", "compare", "transform", "retrieve"}
    required_score = sum(1 for op in required_ops if op in ops)
    optional_score = sum(1 for op in optional_ops if op in ops)
    source_score = min(len(_plan_sources(plan)), 6)
    validation_score = min(len(validation_checks), 4)
    library_cost = 0.0
    reliability_gain = 0.0
    unknown_ops: list[str] = []
    for op in ops:
        spec = get_operator_spec(op)
        if spec is None:
            unknown_ops.append(op)
            library_cost += 1.5
            continue
        library_cost += spec.base_cost
        reliability_gain += spec.reliability_gain

    cost = len(pipeline)
    score = (
        required_score * 3
        + optional_score
        + source_score
        + validation_score * 2
        + reliability_gain
        - library_cost * 0.35
        - len(unknown_ops) * 2
    )
    return {
        "score": round(score, 3),
        "estimated_step_count": cost,
        "estimated_operator_cost": round(library_cost, 3),
        "estimated_reliability_gain": round(reliability_gain, 3),
        "required_ops_covered": sorted(op for op in required_ops if op in ops),
        "optional_ops_covered": sorted(op for op in optional_ops if op in ops),
        "unknown_ops": sorted(set(unknown_ops)),
        "source_count": source_score,
        "validation_count": validation_score,
    }


def _rewrite_plan_to_dag(plan: dict[str, Any]) -> dict[str, Any]:
    pipeline = [step for step in plan.get("pipeline", []) if isinstance(step, dict)]
    nodes: list[dict[str, Any]] = []
    previous_node_id: str | None = None
    produced_by_source: dict[str, str] = {}

    for index, step in enumerate(pipeline, start=1):
        op = str(step.get("op", "Operator"))
        node_id = f"n{index}_{op.lower()}"
        raw_sources = step.get("sources", [])
        if isinstance(raw_sources, str):
            sources = [raw_sources]
        elif isinstance(raw_sources, list):
            sources = [str(source) for source in raw_sources]
        else:
            sources = []

        depends_on: list[str] = []
        for source in sources:
            if source in produced_by_source:
                depends_on.append(produced_by_source[source])

        op_lower = op.lower()
        if not depends_on and previous_node_id is not None and op_lower not in {"scan", "retrieve"}:
            depends_on.append(previous_node_id)

        for source in sources:
            if source and source.lower() not in {"n/a", "none", "unknown"}:
                produced_by_source[source] = node_id

        nodes.append(
            {
                "id": node_id,
                "op": op,
                "goal": step.get("goal", ""),
                "sources": sources,
                "depends_on": sorted(set(depends_on)),
                "success_criteria": step.get("success_criteria", ""),
            }
        )
        previous_node_id = node_id

    stage_by_node: dict[str, int] = {}
    for node in nodes:
        dependencies = node["depends_on"]
        if not dependencies:
            stage_by_node[node["id"]] = 0
        else:
            stage_by_node[node["id"]] = 1 + max(stage_by_node.get(dep, 0) for dep in dependencies)
        node["parallel_stage"] = stage_by_node[node["id"]]

    return {
        "nodes": nodes,
        "parallel_stages": max(stage_by_node.values(), default=-1) + 1,
    }


def _format_plan_for_prompt(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)


def _build_fallback_plan() -> dict[str, Any]:
    plan: dict[str, Any] = {
        "thought": "Use a conservative fallback pipeline because all candidate plan requests failed.",
        "pipeline": [
            {
                "op": "Profile",
                "goal": "Use the structured context profile to identify relevant sources and fields.",
                "sources": ["context profile"],
                "success_criteria": "Relevant files, schemas, and candidate fields are identified.",
            },
            {
                "op": "Scan",
                "goal": "Inspect the most relevant files or schemas with tools.",
                "sources": ["context files"],
                "success_criteria": "The needed records or tables are available for computation.",
            },
            {
                "op": "Filter",
                "goal": "Apply all constraints from the question.",
                "sources": ["observed records"],
                "success_criteria": "Only rows satisfying the question remain.",
            },
            {
                "op": "Extract",
                "goal": "Extract requested attributes with source-like column names.",
                "sources": ["filtered records"],
                "success_criteria": "Final columns and rows match the requested attributes.",
            },
            {
                "op": "Validate",
                "goal": "Check row count, field names, and filter correctness before answering.",
                "sources": ["candidate answer"],
                "success_criteria": "No unsupported rows, renamed fields, or merged attributes remain.",
            },
            {
                "op": "Generate",
                "goal": "Submit the final answer table.",
                "sources": ["validated answer"],
                "success_criteria": "The answer tool receives columns and rows.",
            },
        ],
        "final_answer_contract": {
            "columns": [],
            "row_rules": ["Infer exact columns from the question and observed source fields."],
            "validation_checks": [
                "Confirm every output row satisfies the question.",
                "Prefer exact observed source field names when available.",
                "Do not merge or rename requested fields without evidence.",
            ],
        },
        "_candidate_index": 0,
        "_fallback": True,
    }
    plan["_selection_metrics"] = _score_plan(plan)
    return plan


@dataclass(frozen=True, slots=True)
class MiniAOPAgentConfig(ReActAgentConfig):
    plan_context_depth: int = 4
    profile_sample_rows: int = 3
    profile_max_doc_chars: int = 1200
    candidate_plan_count: int = 3
    answer_review_enabled: bool = True
    parallel_prefetch_enabled: bool = True
    parallel_prefetch_workers: int = 4


class MiniAOPAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: MiniAOPAgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or MiniAOPAgentConfig()
        self.system_prompt = MINI_AOP_SYSTEM_PROMPT

    def _build_plan_messages(
        self,
        task: PublicTask,
        context_listing: dict[str, object],
        candidate_index: int,
    ) -> list[ModelMessage]:
        tool_descriptions = self.tools.describe_for_prompt()
        user_content = (
            f"Question: {task.question}\n\n"
            "Available context profile:\n"
            f"{json.dumps(context_listing, ensure_ascii=False, indent=2)}\n\n"
            "Available tools:\n"
            f"{tool_descriptions}\n\n"
            f"Generate candidate pipeline plan #{candidate_index}. "
            "Use a robust but concise reasoning path, and make the plan meaningfully different if alternatives exist."
        )
        return [
            ModelMessage(role="system", content=MINI_AOP_PLAN_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=build_user_content_with_video(task, user_content),
            ),
        ]

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        plan: dict[str, Any],
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        plan_content = (
            "Selected Mini-AOP pipeline plan to follow:\n"
            f"{_format_plan_for_prompt(plan)}\n\n"
            "DAG rewrite and optimization metadata:\n"
            f"{json.dumps(plan.get('_dag_rewrite', {}), ensure_ascii=False, indent=2)}\n\n"
            "Execute the plan with tools. If observations contradict the plan, adjust cautiously and validate before answering."
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(
                role="user",
                content=build_user_content_with_video(task, build_task_prompt(task)),
            )
        )
        messages.append(ModelMessage(role="user", content=plan_content))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _build_answer_review_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        plan: dict[str, Any],
        proposed_answer: dict[str, Any],
    ) -> list[ModelMessage]:
        recent_steps = state.steps[-8:]
        recent_observations = [
            {
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            }
            for step in recent_steps
        ]
        user_content = (
            "Review the proposed final answer before submission.\n\n"
            f"Question: {task.question}\n\n"
            "Selected Mini-AOP plan:\n"
            f"{_format_plan_for_prompt(plan)}\n\n"
            "Recent tool observations:\n"
            f"{json.dumps(recent_observations, ensure_ascii=False, indent=2)}\n\n"
            "Proposed answer action_input:\n"
            f"{json.dumps(proposed_answer, ensure_ascii=False, indent=2)}\n\n"
            "Return the final answer as exactly one fenced ```json block with keys "
            "`thought`, `action`, and `action_input`. The action must be `answer`. "
            "Keep correct data rows, but fix output columns if they merge fields, rename "
            "observed fields without evidence, or violate the requested attribute granularity. "
            "Return only the attributes requested by the question; do not include explanatory "
            "helper columns such as entity names, IDs, or intermediate scores unless the question "
            "explicitly asks for them. "
            "Do not invent rows or values that are not supported by the observations."
        )
        return [
            ModelMessage(role="system", content=MINI_AOP_SYSTEM_PROMPT),
            ModelMessage(role="user", content=user_content),
        ]

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()

        context_result = self.tools.execute(
            task,
            "list_context",
            {"max_depth": self.config.plan_context_depth},
        )
        context_observation = {
            "ok": context_result.ok,
            "tool": "list_context",
            "content": context_result.content,
        }
        context_raw = (
            "```json\n"
            + json.dumps(
                {
                    "thought": "Mini-AOP begins by linking the available context files.",
                    "action": "list_context",
                    "action_input": {"max_depth": self.config.plan_context_depth},
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        state.steps.append(
            StepRecord(
                step_index=1,
                thought="Mini-AOP begins by linking the available context files.",
                action="list_context",
                action_input={"max_depth": self.config.plan_context_depth},
                raw_response=context_raw,
                observation=context_observation,
                ok=context_result.ok,
            )
        )

        profile_result = self.tools.execute(
            task,
            "profile_context",
            {
                "sample_rows": self.config.profile_sample_rows,
                "max_doc_chars": self.config.profile_max_doc_chars,
            },
        )
        profile_observation = {
            "ok": profile_result.ok,
            "tool": "profile_context",
            "content": profile_result.content,
        }
        profile_raw = (
            "```json\n"
            + json.dumps(
                {
                    "thought": "Mini-AOP builds a structured context profile before planning.",
                    "action": "profile_context",
                    "action_input": {
                        "sample_rows": self.config.profile_sample_rows,
                        "max_doc_chars": self.config.profile_max_doc_chars,
                    },
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        state.steps.append(
            StepRecord(
                step_index=2,
                thought="Mini-AOP builds a structured context profile before planning.",
                action="profile_context",
                action_input={
                    "sample_rows": self.config.profile_sample_rows,
                    "max_doc_chars": self.config.profile_max_doc_chars,
                },
                raw_response=profile_raw,
                observation=profile_observation,
                ok=profile_result.ok,
            )
        )

        candidate_plans: list[dict[str, Any]] = []
        for candidate_index in range(1, self.config.candidate_plan_count + 1):
            try:
                raw_plan = self.model.complete(
                    self._build_plan_messages(task, profile_result.content, candidate_index)
                )
                plan = _parse_plan(raw_plan)
                plan["_candidate_index"] = candidate_index
                plan["_selection_metrics"] = _score_plan(plan)
                candidate_plans.append(plan)
                state.steps.append(
                    StepRecord(
                        step_index=2 + candidate_index,
                        thought=str(plan.get("thought", "Mini-AOP generated a candidate plan.")),
                        action="__mini_aop_candidate_plan__",
                        action_input={"candidate_index": candidate_index, "plan": plan},
                        raw_response=raw_plan,
                        observation={
                            "ok": True,
                            "tool": "__mini_aop_candidate_plan__",
                            "content": {
                                "candidate_index": candidate_index,
                                "selection_metrics": plan["_selection_metrics"],
                                "pipeline_ops": [
                                    step.get("op")
                                    for step in plan.get("pipeline", [])
                                    if isinstance(step, dict)
                                ],
                            },
                        },
                        ok=True,
                    )
                )
            except Exception as exc:
                state.steps.append(
                    StepRecord(
                        step_index=2 + candidate_index,
                        thought="Mini-AOP candidate plan generation failed; trying the next candidate.",
                        action="__mini_aop_candidate_plan_error__",
                        action_input={"candidate_index": candidate_index},
                        raw_response="",
                        observation={
                            "ok": False,
                            "tool": "__mini_aop_candidate_plan_error__",
                            "content": {
                                "candidate_index": candidate_index,
                                "error": str(exc),
                            },
                        },
                        ok=False,
                    )
                )

        try:
            if not candidate_plans:
                fallback_plan = _build_fallback_plan()
                candidate_plans.append(fallback_plan)
                state.steps.append(
                    StepRecord(
                        step_index=3 + self.config.candidate_plan_count,
                        thought="Mini-AOP used a local fallback plan after all candidate plan requests failed.",
                        action="__mini_aop_fallback_plan__",
                        action_input={"plan": fallback_plan},
                        raw_response=json.dumps(fallback_plan, ensure_ascii=False),
                        observation={
                            "ok": True,
                            "tool": "__mini_aop_fallback_plan__",
                            "content": {
                                "selection_metrics": fallback_plan["_selection_metrics"],
                                "pipeline_ops": [
                                    step.get("op")
                                    for step in fallback_plan.get("pipeline", [])
                                    if isinstance(step, dict)
                                ],
                            },
                        },
                        ok=True,
                    )
                )
            plan = max(candidate_plans, key=lambda item: item["_selection_metrics"]["score"])
            plan["_dag_rewrite"] = _rewrite_plan_to_dag(plan)
            next_step_index = 4 + self.config.candidate_plan_count
            state.steps.append(
                StepRecord(
                    step_index=next_step_index,
                    thought="Mini-AOP selected the highest-scoring candidate and rewrote it into a DAG.",
                    action="__mini_aop_select_and_rewrite__",
                    action_input={
                        "selected_candidate_index": plan["_candidate_index"],
                        "selection_metrics": plan["_selection_metrics"],
                        "dag_rewrite": plan["_dag_rewrite"],
                    },
                    raw_response=json.dumps(
                        {
                            "selected_candidate_index": plan["_candidate_index"],
                            "selection_metrics": plan["_selection_metrics"],
                            "dag_rewrite": plan["_dag_rewrite"],
                        },
                        ensure_ascii=False,
                    ),
                    observation={
                        "ok": True,
                        "tool": "__mini_aop_select_and_rewrite__",
                        "content": {
                            "candidate_count": len(candidate_plans),
                            "selected_candidate_index": plan["_candidate_index"],
                            "selection_metrics": plan["_selection_metrics"],
                            "parallel_stages": plan["_dag_rewrite"]["parallel_stages"],
                        },
                    },
                    ok=True,
                )
            )
            next_step_index += 1

            if self.config.parallel_prefetch_enabled:
                prefetch_result = execute_parallel_prefetch(
                    task=task,
                    tools=self.tools,
                    context_profile=profile_result.content,
                    dag_rewrite=plan["_dag_rewrite"],
                    max_workers=self.config.parallel_prefetch_workers,
                )
                state.steps.append(
                    StepRecord(
                        step_index=next_step_index,
                        thought="Mini-AOP executed independent context prefetch tasks in parallel.",
                        action="__mini_aop_parallel_prefetch__",
                        action_input={
                            "max_workers": self.config.parallel_prefetch_workers,
                            "dag_parallel_stages": plan["_dag_rewrite"]["parallel_stages"],
                        },
                        raw_response=json.dumps(prefetch_result, ensure_ascii=False),
                        observation={
                            "ok": True,
                            "tool": "__mini_aop_parallel_prefetch__",
                            "content": prefetch_result,
                        },
                        ok=True,
                    )
                )
                next_step_index += 1
        except Exception as exc:
            state.failure_reason = f"Mini-AOP planning failed: {exc}"
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )

        for _ in range(self.config.max_steps):
            raw_response = self.model.complete(self._build_messages(task, state, plan))
            try:
                model_step = parse_model_step(raw_response)
                if model_step.action == "answer" and self.config.answer_review_enabled:
                    review_observation: dict[str, Any]
                    try:
                        review_raw_response = self.model.complete(
                            self._build_answer_review_messages(
                                task,
                                state,
                                plan,
                                model_step.action_input,
                            )
                        )
                        reviewed_step = parse_model_step(review_raw_response)
                        if reviewed_step.action != "answer":
                            raise ValueError("Mini-AOP answer review must return the answer action.")
                        review_observation = {
                            "ok": True,
                            "tool": "__mini_aop_answer_review__",
                            "content": {
                                "original_action_input": model_step.action_input,
                                "reviewed_action_input": reviewed_step.action_input,
                                "changed": reviewed_step.action_input != model_step.action_input,
                            },
                        }
                        state.steps.append(
                            StepRecord(
                                step_index=next_step_index,
                                thought=reviewed_step.thought,
                                action="__mini_aop_answer_review__",
                                action_input={
                                    "original_action_input": model_step.action_input,
                                    "reviewed_action_input": reviewed_step.action_input,
                                },
                                raw_response=review_raw_response,
                                observation=review_observation,
                                ok=True,
                            )
                        )
                        next_step_index += 1
                        model_step = reviewed_step
                    except Exception as review_exc:
                        state.steps.append(
                            StepRecord(
                                step_index=next_step_index,
                                thought="Mini-AOP answer review failed; using the original answer.",
                                action="__mini_aop_answer_review__",
                                action_input={"original_action_input": model_step.action_input},
                                raw_response="",
                                observation={
                                    "ok": False,
                                    "tool": "__mini_aop_answer_review__",
                                    "error": str(review_exc),
                                },
                                ok=False,
                            )
                        )
                        next_step_index += 1

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(
                    StepRecord(
                        step_index=next_step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )
                next_step_index += 1
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                state.steps.append(
                    StepRecord(
                        step_index=next_step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation={
                            "ok": False,
                            "error": str(exc),
                        },
                        ok=False,
                    )
                )
                next_step_index += 1

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Mini-AOP agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
