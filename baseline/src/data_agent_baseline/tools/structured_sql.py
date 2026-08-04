from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _safe_table_name(raw_name: str, *, fallback: str) -> str:
    candidate = re.sub(r"\W+", "_", raw_name, flags=re.ASCII).strip("_")
    return candidate or fallback


def _load_json_table(path: Path) -> tuple[str, pd.DataFrame] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        table_name = str(payload.get("table") or path.stem)
        return table_name, pd.DataFrame(payload["records"]).convert_dtypes()
    if isinstance(payload, list):
        return path.stem, pd.DataFrame(payload).convert_dtypes()
    return None


@dataclass(frozen=True, slots=True)
class SourceTable:
    name: str
    path: Path
    relative_path: str
    kind: str
    raw_name: str


def _discover_source_tables(context_root: Path) -> list[SourceTable]:
    tables: list[SourceTable] = []
    used_names: set[str] = set()
    for path in sorted(context_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(context_root).as_posix()
        discovered_tables: list[tuple[str, str]] = []
        if path.suffix.lower() == ".json":
            loaded = _load_json_table(path)
            if loaded is not None:
                discovered_tables.append((loaded[0], "json"))
        elif path.suffix.lower() == ".csv":
            discovered_tables.append((path.stem, "csv"))
        elif path.suffix.lower() == ".parquet":
            discovered_tables.append((path.stem, "parquet"))
        elif path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(path) as sqlite_connection:
                table_rows = sqlite_connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
                for (raw_table_name,) in table_rows:
                    discovered_tables.append((str(raw_table_name), "sqlite"))

        for raw_name, kind in discovered_tables:
            base_name = _safe_table_name(
                raw_name,
                fallback=_safe_table_name(path.stem, fallback="data"),
            )
            table_name = base_name
            suffix = 2
            while table_name.lower() in used_names:
                table_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(table_name.lower())
            tables.append(
                SourceTable(
                    name=table_name,
                    path=path,
                    relative_path=relative_path,
                    kind=kind,
                    raw_name=raw_name,
                )
            )
    return tables


def _rewrite_source_references(sql: str, tables: list[SourceTable]) -> str:
    normalized_sql = sql
    for table in tables:
        relative = Path(table.relative_path)
        reference_candidates = [
            table.relative_path,
            table.relative_path.replace("/", "\\"),
            f"{relative.parent.name}.{table.name}",
        ]
        for candidate in reference_candidates:
            normalized_sql = re.sub(
                re.escape(candidate),
                table.name,
                normalized_sql,
                flags=re.IGNORECASE,
            )
    return normalized_sql


def _referenced_tables(sql: str, tables: list[SourceTable]) -> list[SourceTable]:
    return [
        table
        for table in tables
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(table.name)}(?![A-Za-z0-9_])",
            sql,
            flags=re.IGNORECASE,
        )
    ]


def _load_source_table(table: SourceTable) -> pd.DataFrame:
    if table.kind == "json":
        loaded = _load_json_table(table.path)
        if loaded is None:
            raise ValueError(f"JSON source is not tabular: {table.relative_path}")
        return loaded[1]
    if table.kind == "csv":
        return pd.read_csv(table.path).convert_dtypes()
    if table.kind == "parquet":
        return pd.read_parquet(table.path).convert_dtypes()
    if table.kind == "sqlite":
        quoted_name = table.raw_name.replace('"', '""')
        with sqlite3.connect(table.path) as sqlite_connection:
            return pd.read_sql_query(
                f'SELECT * FROM "{quoted_name}"',
                sqlite_connection,
            ).convert_dtypes()
    raise ValueError(f"Unsupported source kind: {table.kind}")


def _canonical_sort_key(row: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    key: list[tuple[int, Any]] = []
    for value in row:
        if value is None or value is pd.NA:
            key.append((2, ""))
        elif isinstance(value, (int, float)):
            key.append((0, float(value)))
        else:
            key.append((1, str(value)))
    return tuple(key)


def rewrite_semantic_column_owners(
    sql: str,
    field_ownership: dict[str, list[str]],
) -> tuple[str, list[dict[str, str]]]:
    if not field_ownership:
        return sql, []

    with duckdb.connect(":memory:") as connection:
        serialized = connection.execute(
            "SELECT json_serialize_sql(?)",
            [sql],
        ).fetchone()[0]
        payload = json.loads(serialized)
        if payload.get("error"):
            return sql, []

        alias_to_table: dict[str, str] = {}

        def collect_tables(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "BASE_TABLE":
                    table_name = str(value.get("table_name", ""))
                    alias = str(value.get("alias") or table_name)
                    if table_name and alias:
                        alias_to_table[alias.casefold()] = table_name
                for child in value.values():
                    collect_tables(child)
            elif isinstance(value, list):
                for child in value:
                    collect_tables(child)

        collect_tables(payload)
        owner_to_alias: dict[str, str] = {}
        for alias, table_name in alias_to_table.items():
            owner_to_alias.setdefault(table_name.casefold(), alias)

        normalized_ownership = {
            field_name.casefold(): owners
            for field_name, owners in field_ownership.items()
            if len(owners) == 1
        }
        corrections: list[dict[str, str]] = []

        def rewrite_columns(value: Any) -> None:
            if isinstance(value, dict):
                column_names = value.get("column_names")
                if (
                    value.get("class") == "COLUMN_REF"
                    and isinstance(column_names, list)
                    and len(column_names) == 2
                ):
                    current_alias = str(column_names[0])
                    field_name = str(column_names[1])
                    owners = normalized_ownership.get(field_name.casefold())
                    if owners:
                        target_alias = owner_to_alias.get(str(owners[0]).casefold())
                        current_table = alias_to_table.get(current_alias.casefold())
                        if (
                            target_alias is not None
                            and current_table is not None
                            and target_alias.casefold() != current_alias.casefold()
                        ):
                            column_names[0] = target_alias
                            corrections.append(
                                {
                                    "field": field_name,
                                    "from": current_alias,
                                    "to": target_alias,
                                    "semantic_owner": str(owners[0]),
                                }
                            )
                for child in value.values():
                    rewrite_columns(child)
            elif isinstance(value, list):
                for child in value:
                    rewrite_columns(child)

        rewrite_columns(payload)
        if not corrections:
            return sql, []
        rewritten_sql = connection.execute(
            "SELECT json_deserialize_sql(?)",
            [json.dumps(payload)],
        ).fetchone()[0]
    return str(rewritten_sql), corrections


def execute_structured_sql(
    context_root: Path,
    sql: str,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "describe", "show", "pragma", "explain")):
        raise ValueError("Only read-only SQL statements are allowed.")

    tables = _discover_source_tables(context_root)
    if not tables:
        raise ValueError("No JSON, CSV, or Parquet tables were found in the task context.")
    normalized_sql = _rewrite_source_references(sql, tables)
    referenced_tables = _referenced_tables(normalized_sql, tables)
    if not referenced_tables:
        available = ", ".join(table.name for table in tables)
        raise ValueError(
            "SQL does not reference a discovered table. "
            f"Available tables: {available}"
        )

    with duckdb.connect(":memory:") as connection:
        table_metadata: list[dict[str, Any]] = []
        for table in referenced_tables:
            frame = _load_source_table(table)
            connection.register(table.name, frame)
            table_metadata.append(
                {
                    "name": table.name,
                    "source": table.relative_path,
                    "columns": [str(column) for column in frame.columns],
                    "row_count": len(frame),
                }
            )
        cursor = connection.execute(normalized_sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    normalized_sql_lower = normalized_sql.lower()
    canonical_order_applied = "order by" not in normalized_sql_lower
    if canonical_order_applied:
        rows = sorted(rows, key=_canonical_sort_key)
    truncated = len(rows) > limit
    return {
        "tables": table_metadata,
        "columns": column_names,
        "rows": [list(row) for row in rows[:limit]],
        "row_count": min(len(rows), limit),
        "truncated": truncated,
        "canonical_order_applied": canonical_order_applied,
        "executed_sql": normalized_sql,
    }
