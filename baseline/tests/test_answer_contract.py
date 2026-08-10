from __future__ import annotations

from data_agent_baseline.agents.answer_contract import (
    apply_answer_contract,
    inspect_answer_contract,
)
from data_agent_baseline.agents.model import ScriptedModelAdapter


def test_contract_detects_missing_temporal_identifier() -> None:
    _, issues, _ = inspect_answer_contract(
        question=(
            "Show the latest record and list total assets, total liabilities, "
            "net foreign assets, and foreign liabilities."
        ),
        answer={
            "columns": ["total assets", "total liabilities"],
            "rows": [[100, 100]],
        },
        steps=[],
    )

    assert "missing_temporal_identifier" in {issue.code for issue in issues}


def test_contract_expands_administrative_names_only_when_requested() -> None:
    answer, _, repairs = inspect_answer_contract(
        question="地区使用完整行政区名称，并列出各地区的 GDP。",
        answer={
            "columns": ["地区", "GDP"],
            "rows": [["北京", 10], ["宁夏回族自治区", 20], ["广州", 30]],
        },
        steps=[],
    )

    assert answer["rows"] == [["北京市", 10], ["宁夏回族自治区", 20], ["广州市", 30]]
    assert len(repairs) == 2


def test_contract_model_review_can_replace_unsupported_unit_conversion() -> None:
    model = ScriptedModelAdapter(
        [
            (
                '{"accept":true,"issues":["unsupported conversion removed"],'
                '"revised_answer":{"columns":["date","amount"],'
                '"rows":[["2004-01-31","7015331"]]}}'
            )
        ]
    )
    run_result = {
        "task_id": "task_unit",
        "answer": {
            "columns": ["date", "amount (yuan)"],
            "rows": [["2004-01-31", "70153310000"]],
        },
        "steps": [
            {
                "step_index": 1,
                "action": "read_doc",
                "raw_response": "",
                "observation": {"unit": "yuan", "value": "7015331"},
                "ok": True,
            },
            {
                "step_index": 2,
                "action": "answer",
                "raw_response": "I assume the unit is ten-thousand yuan and multiply by 10,000.",
                "observation": {"status": "submitted"},
                "ok": True,
            },
        ],
        "failure_reason": None,
        "succeeded": True,
    }

    reviewed = apply_answer_contract(
        question="List the date and amount in yuan.",
        run_result=run_result,
        model=model,
        model_review_enabled=True,
        evidence_max_chars=4000,
    )

    assert reviewed["answer"]["rows"] == [["2004-01-31", "7015331"]]
    assert reviewed["answer_contract"]["model_review_applied"] is True
    assert reviewed["steps"][-1]["action"] == "__answer_contract__"
