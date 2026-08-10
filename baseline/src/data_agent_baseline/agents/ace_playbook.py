from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u3400-\u4dbf\u4e00-\u9fff]+")


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value)
        if len(token) >= 2
    }


def _task_number(task_id: str) -> int:
    return int(task_id.removeprefix("task_"))


def _context_file_types(context_dir: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path.suffix.lower().lstrip(".") or "other"
                for path in context_dir.rglob("*")
                if path.is_file() and not path.name.startswith(".")
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ACEPlaybookEntry:
    entry_id: str
    kind: str
    text: str
    keywords: tuple[str, ...]
    helpful_count: int
    harmful_count: int
    evidence_task_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "text": self.text,
            "keywords": list(self.keywords),
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "evidence_task_ids": list(self.evidence_task_ids),
        }


class ACEPlaybook:
    """Deterministic ACE-lite playbook with incremental delta curation."""

    schema_version = "1.0"

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def _load_payload(self) -> dict[str, Any]:
        if self.path is None or not self.path.is_file():
            return {"schema_version": self.schema_version, "entries": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": self.schema_version, "entries": []}
        if payload.get("schema_version") != self.schema_version:
            return {"schema_version": self.schema_version, "entries": []}
        return payload

    def entries(self) -> list[ACEPlaybookEntry]:
        entries: list[ACEPlaybookEntry] = []
        for raw in self._load_payload().get("entries", []):
            if not isinstance(raw, dict):
                continue
            try:
                entries.append(
                    ACEPlaybookEntry(
                        entry_id=str(raw["entry_id"]),
                        kind=str(raw["kind"]),
                        text=str(raw["text"]),
                        keywords=tuple(str(value) for value in raw.get("keywords", [])),
                        helpful_count=int(raw.get("helpful_count", 0)),
                        harmful_count=int(raw.get("harmful_count", 0)),
                        evidence_task_ids=tuple(
                            str(value) for value in raw.get("evidence_task_ids", [])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return entries

    def retrieve(self, query: str, *, limit: int = 6) -> list[ACEPlaybookEntry]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, ACEPlaybookEntry]] = []
        for entry in self.entries():
            entry_tokens = set(entry.keywords) | _tokens(entry.text)
            union = query_tokens | entry_tokens
            overlap = len(query_tokens & entry_tokens) / len(union) if union else 0.0
            reliability = (entry.helpful_count + 1) / (
                entry.helpful_count + entry.harmful_count + 2
            )
            if overlap > 0:
                scored.append((overlap + reliability * 0.1, entry))
        scored.sort(key=lambda item: (-item[0], item[1].entry_id))
        return [entry for _, entry in scored[:limit]]

    def render_for_prompt(self, query: str, *, limit: int = 6) -> str:
        entries = self.retrieve(query, limit=limit)
        if not entries:
            return ""
        lines = [
            "ACE-lite playbook lessons from disjoint adaptation tasks:",
            "Use them as procedural hints only; never treat them as task answer evidence.",
        ]
        for entry in entries:
            lines.append(
                f"- [{entry.kind}; helpful={entry.helpful_count}; "
                f"harmful={entry.harmful_count}] {entry.text}"
            )
        return "\n".join(lines)

    def apply_deltas(self, deltas: list[dict[str, Any]]) -> Path | None:
        if self.path is None:
            return None
        current = {entry.entry_id: entry.to_dict() for entry in self.entries()}
        for delta in deltas:
            entry_id = str(delta["entry_id"])
            existing = current.get(entry_id)
            if existing is None:
                current[entry_id] = {
                    "entry_id": entry_id,
                    "kind": str(delta["kind"]),
                    "text": str(delta["text"]),
                    "keywords": sorted(
                        {str(value) for value in delta.get("keywords", [])}
                    ),
                    "helpful_count": int(delta.get("helpful_delta", 0)),
                    "harmful_count": int(delta.get("harmful_delta", 0)),
                    "evidence_task_ids": sorted(
                        {str(value) for value in delta.get("evidence_task_ids", [])},
                        key=_task_number,
                    ),
                }
                continue
            existing_task_ids = {str(value) for value in existing.get("evidence_task_ids", [])}
            delta_task_ids = {str(value) for value in delta.get("evidence_task_ids", [])}
            if delta_task_ids - existing_task_ids:
                existing["helpful_count"] = int(existing.get("helpful_count", 0)) + int(
                    delta.get("helpful_delta", 0)
                )
                existing["harmful_count"] = int(existing.get("harmful_count", 0)) + int(
                    delta.get("harmful_delta", 0)
                )
            existing["keywords"] = sorted(
                set(existing.get("keywords", []))
                | {str(value) for value in delta.get("keywords", [])}
            )
            existing["evidence_task_ids"] = sorted(
                existing_task_ids | delta_task_ids,
                key=_task_number,
            )

        ordered = sorted(
            current.values(),
            key=lambda item: (
                str(item["kind"]),
                -int(item["helpful_count"]),
                int(item["harmful_count"]),
                str(item["entry_id"]),
            ),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "method": "ACE-lite deterministic reflection and delta curation",
                    "entries": ordered,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.path


def _trace_actions(trace: dict[str, Any]) -> list[str]:
    return [
        str(step.get("action"))
        for step in trace.get("steps", [])
        if isinstance(step, dict) and step.get("action")
    ]


def reflect_run_to_ace_deltas(
    *,
    run_dir: Path,
    dataset: DABenchPublicDataset,
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reflect task outcomes into answer-free procedural playbook deltas."""
    deltas: list[dict[str, Any]] = []
    for result in evaluation.get("tasks", []):
        if not isinstance(result, dict):
            continue
        task_id = str(result.get("task_id", ""))
        if not task_id:
            continue
        trace_path = run_dir / task_id / "trace.json"
        if not trace_path.is_file():
            continue
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            task = dataset.get_task(task_id)
        except (OSError, json.JSONDecodeError, FileNotFoundError, ValueError):
            continue
        actions = _trace_actions(trace)
        useful_actions = [
            action
            for action in actions
            if action not in {"answer", "__error__", "__forced_answer_error__"}
        ]
        action_counts = Counter(useful_actions)
        repeated = [action for action, count in action_counts.items() if count >= 3]
        file_types = _context_file_types(task.context_dir)
        keywords = sorted(_tokens(task.question) | set(file_types) | set(useful_actions))
        file_label = ", ".join(file_types) or "unknown files"

        if result.get("passed"):
            sequence = " -> ".join(dict.fromkeys(useful_actions)) or "direct grounded answer"
            digest = hashlib.sha256(f"{file_label}|{sequence}".encode()).hexdigest()
            entry_id = "strategy:" + digest[:16]
            deltas.append(
                {
                    "entry_id": entry_id,
                    "kind": "strategy",
                    "text": (
                        f"For workspaces with {file_label}, the verified run used this concise "
                        f"tool pattern: {sequence}. Re-check filters and table granularity before "
                        "submitting."
                    ),
                    "keywords": keywords,
                    "helpful_delta": 1,
                    "harmful_delta": 0,
                    "evidence_task_ids": [task_id],
                }
            )
            continue

        error_code = str(result.get("error_code") or "wrong_answer")
        repeated_text = (
            f" Repeated calls detected: {', '.join(repeated)}."
            if repeated
            else ""
        )
        digest = hashlib.sha256(
            f"{file_label}|{error_code}|{repeated}".encode()
        ).hexdigest()
        entry_id = "pitfall:" + digest[:16]
        deltas.append(
            {
                "entry_id": entry_id,
                "kind": "pitfall",
                "text": (
                    f"For workspaces with {file_label}, avoid ending with {error_code}."
                    f"{repeated_text} Preserve enough budget to validate and submit a complete table."
                ),
                "keywords": keywords,
                "helpful_delta": 0,
                "harmful_delta": 1,
                "evidence_task_ids": [task_id],
            }
        )
    return deltas
def curate_ace_playbook_from_run(
    *,
    run_dir: Path,
    dataset_root: Path,
    evaluation_path: Path,
    playbook_path: Path,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    dataset = DABenchPublicDataset(dataset_root)
    deltas = reflect_run_to_ace_deltas(
        run_dir=run_dir,
        dataset=dataset,
        evaluation=evaluation,
    )
    playbook = ACEPlaybook(playbook_path)
    output = playbook.apply_deltas(deltas)
    return {
        "playbook_path": str(output) if output else None,
        "delta_count": len(deltas),
        "entry_count": len(playbook.entries()),
        "source_run": str(run_dir),
        "source_evaluation": str(evaluation_path),
    }
