from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.agents.agentic_data import (
    AgenticDataLiteAgent,
    build_profile_graph,
    validate_plan_grammar,
)
from data_agent_baseline.agents.agentic_memory import (
    AgenticLongTermMemory,
    bellman_state_values,
)
from data_agent_baseline.agents.agentic_optimizer import optimize_semantic_plan
from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.benchmark.agentic_data_report import (
    analyze_agentic_data_run,
)
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


def _response(payload: dict) -> str:
    return f"```json\n{json.dumps(payload)}\n```"


def _task(tmp_path: Path) -> PublicTask:
    context = tmp_path / "context"
    context.mkdir()
    (context / "patients.csv").write_text(
        "id,status\n1,severe\n2,mild\n",
        encoding="utf-8",
    )
    (context / "knowledge.md").write_text(
        "# Patients\n\nThe patients table uses id to identify each patient.\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id="task_agentic_data",
            difficulty="easy",
            question="List the ids of severe patients.",
        ),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context),
    )


def _valid_plan() -> dict:
    return {
        "thought": "Select the patient table and filter severe rows.",
        "task_decomposition": [
            {
                "id": "task_1",
                "question": "Find severe patients.",
                "dependencies": [],
            }
        ],
        "selected_data": [
            {
                "node_id": "patients.csv",
                "reason": "Contains patient ids and status.",
            }
        ],
        "logical_plan": [
            {
                "id": "op_1",
                "operator": "Scan",
                "inputs": [],
                "source_nodes": ["patients.csv"],
                "instruction": "Scan patients.",
                "output_columns": ["id", "status"],
            },
            {
                "id": "op_2",
                "operator": "Filter",
                "inputs": ["op_1"],
                "source_nodes": ["patients.csv"],
                "instruction": "Keep status severe.",
                "output_columns": ["id"],
            },
            {
                "id": "op_3",
                "operator": "Generate",
                "inputs": ["op_2"],
                "source_nodes": [],
                "instruction": "Return ids.",
                "output_columns": ["id"],
            },
        ],
        "answer_contract": {
            "columns": ["id"],
            "granularity": "One row per severe patient.",
            "checks": ["Apply the severe filter."],
        },
    }


def test_profile_graph_links_sources_with_shared_keywords() -> None:
    profile = {
        "files": [
            {
                "path": "patients.csv",
                "kind": "csv",
                "columns": ["patient_id", "status"],
                "row_count": 2,
            },
            {
                "path": "visits.csv",
                "kind": "csv",
                "columns": ["patient_id", "visit_date"],
                "row_count": 3,
            },
        ]
    }

    graph = build_profile_graph(profile)

    assert graph["node_count"] == 2
    assert graph["edge_count"] == 1
    assert graph["edges"][0]["shared_columns"] == ["patient_id"]


def test_plan_grammar_rejects_forward_dependencies() -> None:
    graph = build_profile_graph(
        {
            "files": [
                {
                    "path": "patients.csv",
                    "kind": "csv",
                    "columns": ["id"],
                    "row_count": 1,
                }
            ]
        }
    )
    plan = _valid_plan()
    plan["logical_plan"][0]["inputs"] = ["op_2"]

    issues = validate_plan_grammar(plan, graph)

    assert any("non-prior input op_2" in issue for issue in issues)


def test_optimizer_adds_costed_physical_plan() -> None:
    graph = build_profile_graph(
        {
            "files": [
                {
                    "path": "patients.csv",
                    "kind": "csv",
                    "columns": ["id", "status"],
                    "row_count": 100,
                }
            ]
        }
    )
    plan = _valid_plan()
    plan["logical_plan"].insert(
        2,
        {
            "id": "op_semantic",
            "operator": "SemanticFilter",
            "inputs": ["op_2"],
            "source_nodes": ["patients.csv"],
            "instruction": "Check semantic eligibility.",
            "output_columns": ["id"],
        },
    )
    plan["logical_plan"][-1]["inputs"] = ["op_semantic"]

    optimized = optimize_semantic_plan(plan, graph)

    assert len(optimized["physical_plan"]) == 4
    assert optimized["optimizer"]["estimated_cost_after"] < optimized["optimizer"][
        "estimated_cost_before"
    ]
    assert "semantic_model_cascade" in optimized["optimizer"]["rules_applied"]


def test_long_term_memory_retrieves_related_high_value_plan(
    tmp_path: Path,
) -> None:
    store = AgenticLongTermMemory(tmp_path / "memory")
    store.store(
        {
            "task_id": "task_1",
            "question": "List severe patients.",
            "kind": "good_plan",
            "state_value": 1.0,
        }
    )
    store.store(
        {
            "task_id": "task_2",
            "question": "Calculate annual revenue.",
            "kind": "good_plan",
            "state_value": 1.0,
        }
    )

    retrieved = store.retrieve("Find severe patient ids.")

    assert len(retrieved) == 1
    assert retrieved[0]["task_id"] == "task_1"
    assert bellman_state_values(
        [{"reward": -1.0}, {"reward": 1.0}],
        gamma=0.5,
    ) == [-0.5, 1.0]


def test_agentic_data_runs_profile_plan_validate_execute_cycle(tmp_path: Path) -> None:
    task = _task(tmp_path)
    model = ScriptedModelAdapter(
        [
            _response(_valid_plan()),
            _response(
                {
                    "accept": True,
                    "issues": [],
                    "transition_feedback": "",
                }
            ),
            _response(
                {
                    "thought": "Submit the grounded severe patient id.",
                    "action": "answer",
                    "action_input": {"columns": ["id"], "rows": [[1]]},
                }
            ),
            _response(
                {
                    "accept": True,
                    "issues": [],
                    "transition_feedback": "",
                }
            ),
        ]
    )

    result = AgenticDataLiteAgent(
        model=model,
        tools=create_default_tool_registry(),
    ).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.columns == ["id"]
    assert result.answer.rows == [[1]]
    actions = [step.action for step in result.steps]
    assert actions[:5] == [
        "__agenticdata_profile_graph__",
        "__agenticdata_memory_retrieval__",
        "__agenticdata_plan__",
        "__agenticdata_grammar_validation__",
        "__agenticdata_semantic_validation__",
    ]
    assert "__agenticdata_plan_optimization__" in actions
    assert "__agenticdata_answer_validation__" in actions


def test_agentic_data_offline_report_summarizes_trace(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_1"
    task_dir.mkdir()
    trace = {
        "task_id": "task_1",
        "succeeded": True,
        "failure_reason": None,
        "steps": [
            {
                "action": "__agenticdata_profile_graph__",
                "ok": True,
                "observation": {
                    "profile_graph": {"node_count": 2, "edge_count": 1}
                },
            },
            {
                "action": "__agenticdata_memory_retrieval__",
                "ok": True,
                "observation": {"retrieved_count": 1},
            },
            {
                "action": "__agenticdata_plan__",
                "ok": True,
                "observation": {},
            },
            {
                "action": "__agenticdata_plan_optimization__",
                "ok": True,
                "observation": {
                    "optimizer": {
                        "estimated_cost_before": 100,
                        "estimated_cost_after": 60,
                        "rules_applied": ["semantic_model_cascade"],
                    }
                },
            },
        ],
    }
    (task_dir / "trace.json").write_text(
        json.dumps(trace),
        encoding="utf-8",
    )

    report = analyze_agentic_data_run(tmp_path)

    assert report["task_count"] == 1
    assert report["succeeded_task_count"] == 1
    assert report["tasks_with_memory_retrieval"] == 1
    assert report["estimated_cost_saving"] == 40.0
