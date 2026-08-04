from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def bellman_state_values(
    transitions: list[dict[str, Any]],
    *,
    gamma: float = 0.95,
) -> list[float]:
    values = [0.0] * len(transitions)
    next_value = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        reward = float(transitions[index].get("reward", 0.0))
        next_value = reward + gamma * next_value
        values[index] = round(next_value, 4)
    return values


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(value.lower())
        if len(token) >= 2
    }


class AgenticLongTermMemory:
    """A small file-backed memory store with lexical retrieval."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def retrieve(self, question: str, *, limit: int = 3) -> list[dict[str, Any]]:
        if self.root is None or not self.root.is_dir():
            return []
        query_tokens = _tokens(question)
        scored: list[tuple[float, dict[str, Any]]] = []
        for path in self.root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            record_tokens = _tokens(str(record.get("question", "")))
            union = query_tokens | record_tokens
            similarity = len(query_tokens & record_tokens) / len(union) if union else 0.0
            value = float(record.get("state_value", 0.0))
            score = similarity + max(value, 0.0) * 0.05
            if similarity > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**record, "retrieval_score": round(score, 4)}
            for score, record in scored[:limit]
        ]

    def store(self, record: dict[str, Any]) -> Path | None:
        if self.root is None:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        record_path = self.root / f"{uuid.uuid4().hex}.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return record_path
