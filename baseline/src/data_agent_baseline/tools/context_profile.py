from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.pdf import extract_pdf_text


def _profile_csv(path: Path, relative_path: str, *, sample_rows: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = []
        for index, row in enumerate(reader):
            rows.append(row)
            if index >= sample_rows:
                break

    header = rows[0] if rows else []
    data_sample = rows[1:]
    row_count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row_count = max(sum(1 for _ in csv.reader(handle)) - 1, 0)

    return {
        "path": relative_path,
        "kind": "csv",
        "columns": header,
        "row_count": row_count,
        "sample_rows": data_sample,
    }


def _summarize_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": list(value.keys())[:20],
            "sample": {
                str(key): _summarize_json_value(child, depth=depth + 1)
                for key, child in list(value.items())[:5]
            },
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_sample": _summarize_json_value(value[0], depth=depth + 1) if value else None,
        }
    return {
        "type": type(value).__name__,
        "sample": str(value)[:120],
    }


def _profile_json(path: Path, relative_path: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _summarize_json_value(payload)
    return {
        "path": relative_path,
        "kind": "json",
        "summary": summary,
    }


def _profile_sqlite(path: Path, relative_path: str) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    with sqlite3.connect(path) as connection:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table_name,) in table_rows:
            columns = [
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            tables.append(
                {
                    "name": table_name,
                    "columns": columns,
                    "row_count": row_count,
                }
            )
    return {
        "path": relative_path,
        "kind": "sqlite",
        "tables": tables,
    }


def _profile_doc(path: Path, relative_path: str, *, max_chars: int) -> dict[str, Any]:
    if path.suffix.lower() == ".pdf":
        text = extract_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": relative_path,
        "kind": "document",
        "char_count": len(text),
        "preview": text[:max_chars],
    }


def profile_context(task: PublicTask, *, sample_rows: int = 3, max_doc_chars: int = 1200) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(task.context_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                files.append(_profile_csv(path, relative_path, sample_rows=sample_rows))
            elif suffix == ".json":
                files.append(_profile_json(path, relative_path))
            elif suffix in {".sqlite", ".sqlite3", ".db"}:
                files.append(_profile_sqlite(path, relative_path))
            elif suffix in {".md", ".pdf", ".txt"}:
                files.append(_profile_doc(path, relative_path, max_chars=max_doc_chars))
            else:
                files.append(
                    {
                        "path": relative_path,
                        "kind": "file",
                        "size": path.stat().st_size,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": relative_path, "error": str(exc)})

    return {
        "root": str(task.context_dir),
        "file_count": len(files),
        "files": files,
        "errors": errors,
    }
