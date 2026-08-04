from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.operators import normalize_operator_name
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


def _context_files(profile: dict[str, Any]) -> list[dict[str, Any]]:
    files = profile.get("files", [])
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict) and isinstance(item.get("path"), str)]


def _prefetch_action(file_profile: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    path = str(file_profile["path"])
    kind = str(file_profile.get("kind", "")).lower()
    suffix = Path(path).suffix.lower()

    if kind == "csv" or suffix == ".csv":
        return "read_csv", {"path": path, "max_rows": 5}
    if kind == "json" or suffix == ".json":
        return "read_json", {"path": path, "max_chars": 2500}
    if kind == "document" or suffix in {".md", ".txt"}:
        return "read_doc", {"path": path, "max_chars": 1800}
    if kind == "sqlite" or suffix in {".db", ".sqlite", ".sqlite3"}:
        return "inspect_sqlite_schema", {"path": path}
    return None


def should_parallel_prefetch(dag_rewrite: dict[str, Any]) -> bool:
    nodes = dag_rewrite.get("nodes", [])
    if not isinstance(nodes, list):
        return False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        op = normalize_operator_name(str(node.get("op", "")))
        if op in {"profile", "scan", "retrieve"} and not node.get("depends_on"):
            return True
    return False


def execute_parallel_prefetch(
    *,
    task: PublicTask,
    tools: ToolRegistry,
    context_profile: dict[str, Any],
    dag_rewrite: dict[str, Any],
    max_workers: int = 4,
) -> dict[str, Any]:
    if not should_parallel_prefetch(dag_rewrite):
        return {
            "enabled": False,
            "reason": "No independent Profile/Scan/Retrieve nodes were found in the DAG.",
            "prefetched": [],
        }

    actions: list[tuple[str, dict[str, Any]]] = []
    for file_profile in _context_files(context_profile):
        action = _prefetch_action(file_profile)
        if action is not None:
            actions.append(action)

    if not actions:
        return {
            "enabled": False,
            "reason": "No supported context files were available for prefetch.",
            "prefetched": [],
        }

    results: list[dict[str, Any]] = []
    worker_count = max(1, min(max_workers, len(actions)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_action = {
            executor.submit(tools.execute, task, action_name, action_input): (
                action_name,
                action_input,
            )
            for action_name, action_input in actions
        }
        for future in as_completed(future_to_action):
            action_name, action_input = future_to_action[future]
            try:
                tool_result = future.result()
                results.append(
                    {
                        "action": action_name,
                        "action_input": action_input,
                        "ok": tool_result.ok,
                        "content": tool_result.content,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "action": action_name,
                        "action_input": action_input,
                        "ok": False,
                        "error": str(exc),
                    }
                )

    results.sort(key=lambda item: (str(item["action_input"].get("path", "")), item["action"]))
    return {
        "enabled": True,
        "max_workers": worker_count,
        "prefetched_count": len(results),
        "prefetched": results,
    }
