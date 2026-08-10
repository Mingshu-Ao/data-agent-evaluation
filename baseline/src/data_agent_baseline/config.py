from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local runtimes.
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    json_mode: bool = False
    ace_playbook_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AnswerContractConfig:
    enabled: bool = False
    model_review_enabled: bool = True
    evidence_max_chars: int = 24000


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    answer_contract: AnswerContractConfig = field(default_factory=AnswerContractConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _secret_value(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    match = ENV_REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        return value
    return os.environ.get(match.group(1), "")


def _parse_scalar(raw_value: str) -> str | int | float | None:
    value = raw_value.strip()
    if not value:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _fallback_safe_load(text: str) -> dict[str, object]:
    payload: dict[str, object] = {}
    current_section: dict[str, object] | None = None

    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line_without_comment.endswith(":"):
            section_name = line_without_comment[:-1].strip()
            current_section = {}
            payload[section_name] = current_section
            continue
        if current_section is not None and ":" in line_without_comment:
            key, raw_value = line_without_comment.split(":", 1)
            current_section[key.strip()] = _parse_scalar(raw_value)
    return payload


def _load_yaml_payload(config_path: Path) -> dict[str, object]:
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _fallback_safe_load(text)


def load_app_config(config_path: Path) -> AppConfig:
    payload = _load_yaml_payload(config_path)
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    contract_defaults = AnswerContractConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    contract_payload = payload.get("answer_contract", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=_secret_value(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        json_mode=bool(agent_payload.get("json_mode", agent_defaults.json_mode)),
        ace_playbook_path=(
            _path_value(str(agent_payload["ace_playbook_path"]), PROJECT_ROOT / "artifacts" / "ace_playbook.json")
            if agent_payload.get("ace_playbook_path")
            else None
        ),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    contract_config = AnswerContractConfig(
        enabled=bool(contract_payload.get("enabled", contract_defaults.enabled)),
        model_review_enabled=bool(
            contract_payload.get(
                "model_review_enabled",
                contract_defaults.model_review_enabled,
            )
        ),
        evidence_max_chars=int(
            contract_payload.get("evidence_max_chars", contract_defaults.evidence_max_chars)
        ),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        run=run_config,
        answer_contract=contract_config,
    )
