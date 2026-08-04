from __future__ import annotations

import copy
from typing import Any

SEMANTIC_OPERATORS = {
    "Generate",
    "SemanticExtract",
    "SemanticFilter",
    "SemanticJoin",
    "SemanticScan",
}

RELATIONAL_SELECTIVITY = {
    "Aggregate": 0.1,
    "Filter": 0.25,
    "Intersect": 0.5,
    "Limit": 0.1,
    "Project": 1.0,
    "Sort": 1.0,
    "Union": 2.0,
    "Validate": 1.0,
}


def _source_cardinality(node: dict[str, Any]) -> float:
    row_count = node.get("row_count")
    if isinstance(row_count, (int, float)) and row_count >= 0:
        return max(float(row_count), 1.0)
    char_count = node.get("char_count")
    if isinstance(char_count, (int, float)) and char_count >= 0:
        return max(float(char_count) / 500.0, 1.0)
    if node.get("kind") == "semi_structured":
        return 100.0
    return 25.0


def _estimate_step(
    step: dict[str, Any],
    *,
    source_sizes: dict[str, float],
    output_sizes: dict[str, float],
) -> tuple[float, float]:
    source_nodes = [str(item) for item in step.get("source_nodes", [])]
    input_ids = [str(item) for item in step.get("inputs", [])]
    source_size = sum(source_sizes.get(item, 25.0) for item in source_nodes)
    input_size = sum(output_sizes.get(item, 0.0) for item in input_ids)
    cardinality = max(source_size, input_size, 1.0)
    operator = str(step.get("operator", ""))

    if operator == "Join":
        cardinality = max(cardinality * 0.5, 1.0)
    else:
        cardinality = max(cardinality * RELATIONAL_SELECTIVITY.get(operator, 1.0), 1.0)

    unit_cost = 8.0 if operator in SEMANTIC_OPERATORS else 1.0
    return cardinality, cardinality * unit_cost


def optimize_semantic_plan(
    plan: dict[str, Any],
    profile_graph: dict[str, Any],
) -> dict[str, Any]:
    """Build a cost-annotated physical plan without changing logical dependencies."""
    optimized_plan = copy.deepcopy(plan)
    nodes = [
        node for node in profile_graph.get("nodes", []) if isinstance(node, dict)
    ]
    source_sizes = {
        str(node.get("node_id")): _source_cardinality(node)
        for node in nodes
        if node.get("node_id") is not None
    }

    logical_steps = [
        step for step in optimized_plan.get("logical_plan", []) if isinstance(step, dict)
    ]
    referenced_sources = {
        str(node_id)
        for step in logical_steps
        for node_id in step.get("source_nodes", [])
    }
    selected_data = [
        item for item in optimized_plan.get("selected_data", []) if isinstance(item, dict)
    ]
    pruned_selected_data = [
        item for item in selected_data if str(item.get("node_id")) in referenced_sources
    ]

    rules_applied: list[str] = []
    if pruned_selected_data and len(pruned_selected_data) < len(selected_data):
        optimized_plan["selected_data"] = pruned_selected_data
        rules_applied.append("prune_unreferenced_sources")

    physical_steps: list[dict[str, Any]] = []
    output_sizes: dict[str, float] = {}
    estimated_cost_before = 0.0
    estimated_cost_after = 0.0
    semantic_step_count = 0

    for step in logical_steps:
        cardinality, baseline_cost = _estimate_step(
            step,
            source_sizes=source_sizes,
            output_sizes=output_sizes,
        )
        step_id = str(step.get("id", f"step_{len(physical_steps) + 1}"))
        operator = str(step.get("operator", ""))
        is_semantic = operator in SEMANTIC_OPERATORS
        optimized_cost = baseline_cost * 0.45 if is_semantic else baseline_cost
        backend = "llm_cascade" if is_semantic else "duckdb_or_native"
        strategy = (
            ["embedding_filter", "small_llm", "large_llm_fallback"]
            if is_semantic
            else ["predicate_pushdown", "column_pruning"]
        )
        if is_semantic:
            semantic_step_count += 1

        physical_steps.append(
            {
                **step,
                "backend": backend,
                "strategy": strategy,
                "estimated_input_cardinality": round(cardinality, 3),
                "estimated_cost_before": round(baseline_cost, 3),
                "estimated_cost_after": round(optimized_cost, 3),
            }
        )
        output_sizes[step_id] = cardinality
        estimated_cost_before += baseline_cost
        estimated_cost_after += optimized_cost

    if semantic_step_count:
        rules_applied.append("semantic_model_cascade")
    if any(
        str(step.get("operator")) in {"Filter", "Project"}
        for step in logical_steps
    ):
        rules_applied.append("relational_pushdown_hint")

    optimized_plan["physical_plan"] = physical_steps
    optimized_plan["optimizer"] = {
        "model": "agenticdata_lite_proxy_cost_v1",
        "source_cardinality": {
            key: round(value, 3) for key, value in source_sizes.items()
        },
        "estimated_cost_before": round(estimated_cost_before, 3),
        "estimated_cost_after": round(estimated_cost_after, 3),
        "estimated_saving": round(
            max(estimated_cost_before - estimated_cost_after, 0.0),
            3,
        ),
        "rules_applied": rules_applied,
        "note": (
            "Proxy cost uses source cardinality and operator weights; it is not the "
            "paper's production cost model."
        ),
    }
    return optimized_plan
