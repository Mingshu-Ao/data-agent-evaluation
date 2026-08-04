from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from data_agent_baseline.agents.dagent import (
    DAgentLiteAgent,
    _extract_semantic_field_ownership,
    _retrieve_relevant_document_evidence,
)
from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry
from data_agent_baseline.tools.structured_sql import rewrite_semantic_column_owners


def _json_response(payload: str) -> str:
    return f"```json\n{payload}\n```"


def _plan_response() -> str:
    return _json_response(
        """
{
  "thought": "Retrieve the requested records and verify the output.",
  "decomposition_required": false,
  "sub_questions": [
    {
      "id": "q1",
      "question": "Find the requested records.",
      "retrieval_strategy": "hybrid",
      "expected_evidence": "Matching source rows"
    }
  ],
  "sql_rewrite_policy": {"enabled": true, "goal": "Keep SQL relevant."},
  "final_answer_contract": {
    "columns": ["id", "name"],
    "row_rules": "Only matching rows.",
    "validation_checks": ["Check filters."]
  }
}
""".strip()
    )


def _report_response() -> str:
    return _json_response(
        """
{
  "title": "Severe Cases",
  "summary": "The requested records were retrieved.",
  "findings": ["One matching row was found."],
  "caveats": []
}
""".strip()
    )


def _review_response(*, accept: bool, issue: str = "") -> str:
    issues = f'["{issue}"]' if issue else "[]"
    suggested_action = "Query the semantic owner." if issue else ""
    return _json_response(
        "{\n"
        f'  "accept": {str(accept).lower()},\n'
        f'  "issues": {issues},\n'
        f'  "suggested_action": "{suggested_action}"\n'
        "}"
    )


def _make_task(tmp_path: Path) -> PublicTask:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    return PublicTask(
        record=TaskRecord(
            task_id="task_test",
            difficulty="easy",
            question="List severe cases.",
        ),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )


def test_model_step_parser_accepts_unclosed_json_fence() -> None:
    step = parse_model_step(
        """
```json
{
  "thought": "Inspect context.",
  "action": "list_context",
  "action_input": {"max_depth": 2}
}
""".strip()
    )

    assert step.action == "list_context"
    assert step.action_input == {"max_depth": 2}


def test_dagent_plans_uses_tool_answers_and_generates_report(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "patients.csv").write_text(
        "id,name,severity\n1,Alice,severe\n",
        encoding="utf-8",
    )
    model = ScriptedModelAdapter(
        [
            _plan_response(),
            _json_response(
                """
{
  "thought": "Inspect available context.",
  "action": "list_context",
  "action_input": {"max_depth": 2}
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "Submit the grounded result.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, "Alice"]]}
}
""".strip()
            ),
            _review_response(accept=True),
            _report_response(),
        ]
    )

    result = DAgentLiteAgent(model=model, tools=create_default_tool_registry()).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [[1, "Alice"]]
    actions = [step.action for step in result.steps]
    assert actions == [
        "__dagent_context_profile__",
        "__dagent_plan__",
        "list_context",
        "answer",
        "__dagent_answer_review__",
        "__dagent_report_generation__",
    ]
    report_step = result.steps[-1]
    assert "# Severe Cases" in report_step.observation["report_markdown"]


def test_dagent_rewrites_sql_before_execution(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "patients.json").write_text(
        json.dumps(
            {
                "table": "patients",
                "records": [
                    {"id": 1, "name": "Alice", "severity": "severe"},
                    {"id": 2, "name": "Bob", "severity": "mild"},
                ],
            }
        ),
        encoding="utf-8",
    )

    model = ScriptedModelAdapter(
        [
            _plan_response(),
            _json_response(
                """
{
  "thought": "Query the relational source.",
  "action": "execute_structured_sql",
  "action_input": {"sql": "SELECT * FROM patients"}
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "Filter and project only relevant fields.",
  "keep_original": false,
  "sql": "SELECT id, name FROM patients WHERE severity = 'severe'"
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "Submit the SQL result.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, "Alice"]]}
}
""".strip()
            ),
            _review_response(accept=True),
            _report_response(),
        ]
    )

    result = DAgentLiteAgent(model=model, tools=create_default_tool_registry()).run(task)

    assert result.succeeded
    rewrite_step = next(
        step for step in result.steps if step.action == "__dagent_sql_rewrite__"
    )
    assert rewrite_step.observation["final_sql"].endswith("severity = 'severe'")
    sql_step = next(step for step in result.steps if step.action == "execute_structured_sql")
    assert sql_step.observation["content"]["rows"] == [[1, "Alice"]]
    assert sql_step.action_input["sql"].endswith("severity = 'severe'")


def test_dagent_retrieves_question_relevant_knowledge_excerpt(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text(
        (
            "# Guide\n"
            "General background.\n\n"
            "## Use Case: Severe Thrombosis\n"
            "SELECT ID FROM Examination WHERE Thrombosis = 2\n"
            "This level indicates severe cases."
        ),
        encoding="utf-8",
    )

    evidence = _retrieve_relevant_document_evidence(task)

    assert evidence
    assert "Thrombosis = 2" in evidence[0]["text"]


def test_dagent_extracts_semantic_field_ownership(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text(
        (
            "## Core Entities\n"
            "### Patient\n"
            "- **ID (integer):** Identifier.\n"
            "- **Diagnosis (text):** Disease diagnosed in the patient.\n"
            "### Examination\n"
            "- **ID (integer):** Patient identifier.\n"
            "- **Thrombosis (integer):** Severity.\n"
        ),
        encoding="utf-8",
    )

    ownership = _extract_semantic_field_ownership(task)

    assert ownership["Diagnosis"] == ["Patient"]
    assert ownership["ID"] == ["Examination", "Patient"]


def test_dagent_preserves_diagnostic_sql_follow_up(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "patients.json").write_text(
        json.dumps(
            {
                "table": "patients",
                "records": [
                    {"id": 1, "name": "Alice", "severity": "severe"},
                    {"id": 2, "name": "Bob", "severity": "mild"},
                ],
            }
        ),
        encoding="utf-8",
    )
    model = ScriptedModelAdapter(
        [
            _plan_response(),
            _json_response(
                """
{
  "thought": "Retrieve severe records.",
  "action": "execute_structured_sql",
  "action_input": {"sql": "SELECT id, name FROM patients WHERE severity = 'severe'"}
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "The main query is correct.",
  "keep_original": true,
  "sql": "SELECT id, name FROM patients WHERE severity = 'severe'"
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "Validate the first record.",
  "action": "execute_structured_sql",
  "action_input": {"sql": "SELECT id, name FROM patients WHERE id = 1"}
}
""".strip()
            ),
            _json_response(
                """
{
  "thought": "Submit the validated result.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, "Alice"]]}
}
""".strip()
            ),
            _review_response(accept=True),
            _report_response(),
        ]
    )

    result = DAgentLiteAgent(model=model, tools=create_default_tool_registry()).run(task)

    assert result.succeeded
    sql_steps = [
        step for step in result.steps if step.action == "execute_structured_sql"
    ]
    assert len(sql_steps) == 2
    assert sql_steps[-1].action_input["sql"].endswith("id = 1")


def test_structured_sql_canonicalizes_only_unordered_results(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "values.json").write_text(
        json.dumps(
            {
                "table": "values",
                "records": [{"id": 3}, {"id": 1}, {"id": 2}],
            }
        ),
        encoding="utf-8",
    )
    tools = create_default_tool_registry()

    unordered = tools.execute(
        task,
        "execute_structured_sql",
        {"sql": "SELECT id FROM values"},
    )
    descending = tools.execute(
        task,
        "execute_structured_sql",
        {"sql": "SELECT id FROM values ORDER BY id DESC"},
    )

    assert unordered.content["rows"] == [[1], [2], [3]]
    assert unordered.content["canonical_order_applied"] is True
    assert descending.content["rows"] == [[3], [2], [1]]
    assert descending.content["canonical_order_applied"] is False


def test_structured_sql_does_not_load_unreferenced_csv(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "values.json").write_text(
        json.dumps(
            {
                "table": "values",
                "records": [{"id": 1}],
            }
        ),
        encoding="utf-8",
    )
    (task.context_dir / "unused.csv").write_bytes(b"\xff\xfe\x00\x00")
    tools = create_default_tool_registry()

    result = tools.execute(
        task,
        "execute_structured_sql",
        {"sql": "SELECT id FROM values"},
    )

    assert result.content["rows"] == [[1]]
    assert [table["name"] for table in result.content["tables"]] == ["values"]


def test_structured_sql_joins_sqlite_json_and_csv_sources(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    db_dir = task.context_dir / "db"
    json_dir = task.context_dir / "json"
    csv_dir = task.context_dir / "csv"
    db_dir.mkdir()
    json_dir.mkdir()
    csv_dir.mkdir()
    with sqlite3.connect(db_dir / "event.db") as connection:
        connection.execute("CREATE TABLE event (event_id TEXT, event_name TEXT)")
        connection.execute("INSERT INTO event VALUES ('event-1', 'October Meeting')")
    (json_dir / "budget.json").write_text(
        json.dumps(
            {
                "table": "budget",
                "records": [{"budget_id": "budget-1", "link_to_event": "event-1"}],
            }
        ),
        encoding="utf-8",
    )
    (csv_dir / "expense.csv").write_text(
        "expense_description,cost,approved,link_to_budget\n"
        "Food,175.39,true,budget-1\n",
        encoding="utf-8",
    )
    tools = create_default_tool_registry()

    result = tools.execute(
        task,
        "execute_structured_sql",
        {
            "sql": (
                "SELECT ev.event_name, SUM(ex.cost) AS total "
                "FROM db/event.db ev "
                "JOIN json/budget.json b ON ev.event_id = b.link_to_event "
                "JOIN csv/expense.csv ex ON b.budget_id = ex.link_to_budget "
                "WHERE ex.approved = true GROUP BY ev.event_name"
            )
        },
    )

    assert result.content["rows"] == [["October Meeting", 175.39]]
    assert "db/event.db" not in result.content["executed_sql"]
    assert "json/budget.json" not in result.content["executed_sql"]
    assert "csv/expense.csv" not in result.content["executed_sql"]


def test_semantic_sql_rewrite_corrects_unique_field_owner() -> None:
    sql = (
        "SELECT e.ID, p.SEX, e.Diagnosis "
        "FROM Examination e JOIN Patient p ON e.ID = p.ID "
        "WHERE e.Thrombosis = 2"
    )

    rewritten, corrections = rewrite_semantic_column_owners(
        sql,
        {
            "ID": ["Examination", "Patient"],
            "SEX": ["Patient"],
            "Diagnosis": ["Patient"],
            "Thrombosis": ["Examination"],
        },
    )

    assert "p.Diagnosis" in rewritten
    assert "e.ID" in rewritten
    assert corrections == [
        {
            "field": "Diagnosis",
            "from": "e",
            "to": "p",
            "semantic_owner": "Patient",
        }
    ]


def test_dagent_retries_after_answer_review_rejection(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "patients.csv").write_text(
        "id,name\n1,Alice\n",
        encoding="utf-8",
    )
    model = ScriptedModelAdapter(
        [
            _plan_response(),
            _json_response(
                """
{
  "thought": "Submit an incomplete answer.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, null]]}
}
""".strip()
            ),
            _review_response(accept=False, issue="The requested name is null."),
            _json_response(
                """
{
  "thought": "Submit the corrected answer.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, "Alice"]]}
}
""".strip()
            ),
            _review_response(accept=True),
            _report_response(),
        ]
    )

    result = DAgentLiteAgent(model=model, tools=create_default_tool_registry()).run(task)

    assert result.succeeded
    assert result.answer is not None
    assert result.answer.rows == [[1, "Alice"]]
    reviews = [
        step for step in result.steps if step.action == "__dagent_answer_review__"
    ]
    assert [step.observation["accept"] for step in reviews] == [False, True]


def test_dagent_accepts_consistent_no_action_review(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "patients.csv").write_text(
        "id,name\n1,Alice\n",
        encoding="utf-8",
    )
    model = ScriptedModelAdapter(
        [
            _plan_response(),
            _json_response(
                """
{
  "thought": "Submit the grounded answer.",
  "action": "answer",
  "action_input": {"columns": ["id", "name"], "rows": [[1, "Alice"]]}
}
""".strip()
            ),
            _json_response(
                """
{
  "accept": false,
  "issues": ["The answer is correct."],
  "suggested_action": "No further action needed."
}
""".strip()
            ),
            _report_response(),
        ]
    )

    result = DAgentLiteAgent(model=model, tools=create_default_tool_registry()).run(task)

    assert result.succeeded
    review = next(
        step for step in result.steps if step.action == "__dagent_answer_review__"
    )
    assert review.observation["accept"] is True
    assert review.observation["consistency_override"] is True
