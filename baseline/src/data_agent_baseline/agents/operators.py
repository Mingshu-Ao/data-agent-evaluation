from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    purpose: str
    expected_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    base_cost: float
    reliability_gain: float


OPERATOR_LIBRARY: dict[str, OperatorSpec] = {
    "plan": OperatorSpec(
        name="Plan",
        purpose="Decompose the question into executable data operations.",
        expected_inputs=("question", "context_listing"),
        expected_outputs=("pipeline",),
        base_cost=1.0,
        reliability_gain=1.5,
    ),
    "profile": OperatorSpec(
        name="Profile",
        purpose="Summarize context schemas, file types, and key fields before planning.",
        expected_inputs=("context",),
        expected_outputs=("context_profile",),
        base_cost=0.8,
        reliability_gain=1.4,
    ),
    "scan": OperatorSpec(
        name="Scan",
        purpose="Inspect available files, schemas, and previews.",
        expected_inputs=("context",),
        expected_outputs=("source_inventory",),
        base_cost=1.0,
        reliability_gain=1.2,
    ),
    "retrieve": OperatorSpec(
        name="Retrieve",
        purpose="Load relevant records or document snippets.",
        expected_inputs=("source_inventory",),
        expected_outputs=("candidate_records",),
        base_cost=1.4,
        reliability_gain=1.2,
    ),
    "link": OperatorSpec(
        name="Link",
        purpose="Join or align entities across files.",
        expected_inputs=("candidate_records",),
        expected_outputs=("linked_records",),
        base_cost=1.7,
        reliability_gain=1.6,
    ),
    "filter": OperatorSpec(
        name="Filter",
        purpose="Keep rows satisfying constraints from the question.",
        expected_inputs=("records", "question_constraints"),
        expected_outputs=("filtered_records",),
        base_cost=1.2,
        reliability_gain=1.4,
    ),
    "extract": OperatorSpec(
        name="Extract",
        purpose="Select requested attributes without changing their granularity.",
        expected_inputs=("filtered_records", "answer_contract"),
        expected_outputs=("answer_columns", "answer_rows"),
        base_cost=1.1,
        reliability_gain=1.4,
    ),
    "transform": OperatorSpec(
        name="Transform",
        purpose="Normalize values only when required by the question.",
        expected_inputs=("records",),
        expected_outputs=("normalized_records",),
        base_cost=1.3,
        reliability_gain=0.9,
    ),
    "groupby": OperatorSpec(
        name="GroupBy",
        purpose="Partition rows for grouped aggregation.",
        expected_inputs=("records", "group_keys"),
        expected_outputs=("groups",),
        base_cost=1.4,
        reliability_gain=1.1,
    ),
    "aggregate": OperatorSpec(
        name="Aggregate",
        purpose="Compute counts, sums, averages, minima, or maxima.",
        expected_inputs=("groups",),
        expected_outputs=("aggregate_values",),
        base_cost=1.3,
        reliability_gain=1.2,
    ),
    "compare": OperatorSpec(
        name="Compare",
        purpose="Rank, compare, or select extrema.",
        expected_inputs=("records",),
        expected_outputs=("selected_records",),
        base_cost=1.1,
        reliability_gain=1.0,
    ),
    "validate": OperatorSpec(
        name="Validate",
        purpose="Check filters, row counts, columns, and unsupported renames before answering.",
        expected_inputs=("answer_columns", "answer_rows", "observations"),
        expected_outputs=("validated_answer",),
        base_cost=1.0,
        reliability_gain=2.0,
    ),
    "generate": OperatorSpec(
        name="Generate",
        purpose="Submit the final answer table.",
        expected_inputs=("validated_answer",),
        expected_outputs=("answer",),
        base_cost=0.7,
        reliability_gain=0.8,
    ),
}


def normalize_operator_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "read": "retrieve",
        "load": "retrieve",
        "join": "link",
        "select": "extract",
        "summarize": "aggregate",
        "check": "validate",
        "answer": "generate",
    }
    return aliases.get(normalized, normalized)


def get_operator_spec(name: str) -> OperatorSpec | None:
    return OPERATOR_LIBRARY.get(normalize_operator_name(name))
