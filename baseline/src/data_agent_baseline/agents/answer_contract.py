from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

ANSWER_CONTRACT_REVIEW_PROMPT = """
You are a gold-independent final-answer contract reviewer for a data agent.

Review the proposed table only against the user question and evidence observed by tools. Never use or
request a gold answer. Preserve source values, row order, spelling, and granularity unless the evidence
clearly supports a correction.

Rules:
1. Include every identifier needed to identify or order the requested records, especially a date for
   latest-record or date-ordered questions.
2. Preserve full entity names when the question explicitly requests them.
3. Do not infer a unit from a different table or from outside knowledge. Apply a numeric unit conversion
   only when evidence tied to the requested source explicitly states both the source unit and the needed
   conversion. If the source already states the requested unit, keep the disclosed numeric values.
4. Every row must have exactly the same number of values as the columns.
5. Do not invent rows, columns, dates, or values that are absent from the evidence.

Return one JSON object with:
- accept: boolean
- issues: list of concise strings
- revised_answer: null when unchanged, otherwise an object with columns and rows

Return no text outside the JSON object.
""".strip()


ADMINISTRATIVE_NAME_MAP = {
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
    "河北": "河北省",
    "山西": "山西省",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "海南": "海南省",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "广州": "广州市",
    "深圳": "深圳市",
}

TEMPORAL_QUESTION_PATTERN = re.compile(
    r"(日期|报告期|截止时间|截止日期|期末|按.*(?:升序|降序)|最新|最早|\bdate\b|"
    r"\blatest\b|\bearliest\b|\bchronological\b)",
    re.IGNORECASE,
)
TEMPORAL_COLUMN_PATTERN = re.compile(
    r"(日期|报告期|截止|期末|时间|年份|年月|\bdate\b|\btime\b|\byear\b|\bperiod\b)",
    re.IGNORECASE,
)
FULL_ADMIN_PATTERN = re.compile(
    r"(完整行政区名称|完整(?:地区|省市)名称|full administrative name)",
    re.IGNORECASE,
)
LOCATION_COLUMN_PATTERN = re.compile(
    r"(地区|省市|行政区|region|province|city|location)",
    re.IGNORECASE,
)
UNIT_ASSUMPTION_PATTERN = re.compile(
    r"(assum(?:e|ed|ing)|based on|假设|推测|我认为|可能|据此认为)",
    re.IGNORECASE,
)
UNIT_CONVERSION_PATTERN = re.compile(
    r"(unit|convert|conversion|单位|换算|乘以|multiply|万元|亿元|10[,.]?000)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ContractIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


def _copy_answer(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in answer.get("columns", [])],
        "rows": [list(row) for row in answer.get("rows", []) if isinstance(row, (list, tuple))],
    }


def _normalize_full_administrative_names(
    question: str,
    answer: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = _copy_answer(answer)
    repairs: list[dict[str, Any]] = []
    if not FULL_ADMIN_PATTERN.search(question):
        return normalized, repairs

    location_indexes = [
        index
        for index, column in enumerate(normalized["columns"])
        if LOCATION_COLUMN_PATTERN.search(column)
    ]
    for row_index, row in enumerate(normalized["rows"]):
        for column_index in location_indexes:
            if column_index >= len(row):
                continue
            original = str(row[column_index]).strip()
            replacement = ADMINISTRATIVE_NAME_MAP.get(original)
            if replacement is None:
                continue
            row[column_index] = replacement
            repairs.append(
                {
                    "code": "expand_administrative_name",
                    "row_index": row_index,
                    "column": normalized["columns"][column_index],
                    "before": original,
                    "after": replacement,
                }
            )
    return normalized, repairs


def inspect_answer_contract(
    *,
    question: str,
    answer: dict[str, Any],
    steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[ContractIssue], list[dict[str, Any]]]:
    normalized, repairs = _normalize_full_administrative_names(question, answer)
    columns = normalized["columns"]
    rows = normalized["rows"]
    issues: list[ContractIssue] = []

    if not columns:
        issues.append(ContractIssue("missing_columns", "The submitted table has no columns."))
    if len(set(columns)) != len(columns):
        issues.append(ContractIssue("duplicate_columns", "The submitted table has duplicate columns."))
    for row_index, row in enumerate(rows):
        if len(row) != len(columns):
            issues.append(
                ContractIssue(
                    "row_width_mismatch",
                    f"Row {row_index} has {len(row)} values for {len(columns)} columns.",
                )
            )

    if TEMPORAL_QUESTION_PATTERN.search(question) and not any(
        TEMPORAL_COLUMN_PATTERN.search(column) for column in columns
    ):
        issues.append(
            ContractIssue(
                "missing_temporal_identifier",
                "The question identifies or orders records by time, but the output has no time column.",
            )
        )

    answer_responses = [
        str(step.get("raw_response", ""))
        for step in steps
        if isinstance(step, dict) and step.get("action") == "answer"
    ]
    if any(
        UNIT_ASSUMPTION_PATTERN.search(response) and UNIT_CONVERSION_PATTERN.search(response)
        for response in answer_responses
    ):
        issues.append(
            ContractIssue(
                "unsupported_unit_conversion",
                "The answer appears to apply a unit conversion based on an explicit assumption.",
            )
        )

    return normalized, issues, repairs


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        text = text.removesuffix("```")
    return text.strip()


def _parse_review(raw_response: str) -> dict[str, Any]:
    payload = json.loads(_strip_json_fence(raw_response))
    if not isinstance(payload, dict):
        raise TypeError("Answer contract review must be a JSON object.")
    return payload


def _compact_evidence(steps: list[dict[str, Any]], max_chars: int) -> str:
    rendered_steps: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or not step.get("ok") or step.get("action") == "answer":
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        rendered = json.dumps(
            {
                "step_index": step.get("step_index"),
                "action": step.get("action"),
                "action_input": step.get("action_input", {}),
                "observation": observation,
            },
            ensure_ascii=False,
            default=str,
        )
        if len(rendered) > 8000:
            rendered = rendered[:3500] + "...[middle omitted]..." + rendered[-4500:]
        rendered_steps.append(rendered)

    evidence = "\n".join(rendered_steps)
    if len(evidence) <= max_chars:
        return evidence
    head_size = max_chars // 3
    tail_size = max_chars - head_size
    return evidence[:head_size] + "\n...[evidence omitted]...\n" + evidence[-tail_size:]


def _valid_revised_answer(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not columns or not isinstance(rows, list):
        return None
    if not all(isinstance(row, list) and len(row) == len(columns) for row in rows):
        return None
    return {
        "columns": [str(column) for column in columns],
        "rows": [list(row) for row in rows],
    }


def apply_answer_contract(
    *,
    question: str,
    run_result: dict[str, Any],
    model: ModelAdapter,
    model_review_enabled: bool,
    evidence_max_chars: int,
) -> dict[str, Any]:
    answer = run_result.get("answer")
    if not isinstance(answer, dict):
        return run_result

    steps = run_result.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    checked_answer, issues, repairs = inspect_answer_contract(
        question=question,
        answer=answer,
        steps=steps,
    )
    report: dict[str, Any] = {
        "enabled": True,
        "gold_independent": True,
        "initial_issues": [issue.to_dict() for issue in issues],
        "deterministic_repairs": repairs,
        "model_review_attempted": False,
        "model_review_applied": False,
        "model_review_issues": [],
        "validator_error": None,
    }

    final_answer = checked_answer
    raw_review = ""
    if model_review_enabled:
        report["model_review_attempted"] = True
        try:
            raw_review = model.complete(
                [
                    ModelMessage(role="system", content=ANSWER_CONTRACT_REVIEW_PROMPT),
                    ModelMessage(
                        role="user",
                        content=(
                            f"Question:\n{question}\n\n"
                            f"Deterministic issues:\n"
                            f"{json.dumps(report['initial_issues'], ensure_ascii=False)}\n\n"
                            f"Proposed answer:\n{json.dumps(final_answer, ensure_ascii=False)}\n\n"
                            f"Observed evidence:\n{_compact_evidence(steps, evidence_max_chars)}"
                        ),
                    ),
                ]
            )
            review = _parse_review(raw_review)
            review_issues = review.get("issues", [])
            if isinstance(review_issues, list):
                report["model_review_issues"] = [str(issue) for issue in review_issues]
            revised_answer = _valid_revised_answer(review.get("revised_answer"))
            if revised_answer is not None and revised_answer != final_answer:
                final_answer = revised_answer
                report["model_review_applied"] = True
        except Exception as exc:  # noqa: BLE001
            report["validator_error"] = str(exc)

    run_result["answer"] = final_answer
    run_result["answer_contract"] = report
    run_result.setdefault("steps", []).append(
        {
            "step_index": len(steps) + 1,
            "thought": "Validate the final table against the question and observed evidence.",
            "action": "__answer_contract__",
            "action_input": {},
            "raw_response": raw_review,
            "observation": report,
            "ok": report["validator_error"] is None,
        }
    )
    return run_result
