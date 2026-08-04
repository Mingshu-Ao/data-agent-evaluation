from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from time import monotonic, perf_counter
from typing import Any

from data_agent_baseline.agents.agentic_data import (
    AgenticDataLiteAgent,
    AgenticDataLiteConfig,
)
from data_agent_baseline.agents.dagent import DAgentLiteAgent, DAgentLiteConfig
from data_agent_baseline.agents.mini_aop import MiniAOPAgent, MiniAOPAgentConfig
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    analysis_report_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "analysis_report_path": (
                str(self.analysis_report_path) if self.analysis_report_path else None
            ),
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    agent_kind: str = "react",
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    effective_model = model or build_model_adapter(config)
    effective_tools = tools or create_default_tool_registry()
    if agent_kind.startswith("agentic_data"):
        agent = AgenticDataLiteAgent(
            model=effective_model,
            tools=effective_tools,
            config=AgenticDataLiteConfig(
                max_steps=config.agent.max_steps,
                answer_validation_enabled=agent_kind != "agentic_data_no_validation",
                profile_enabled=agent_kind != "agentic_data_no_profile",
                edge_profile_enabled=agent_kind != "agentic_data_no_edge_profile",
                plan_validation_enabled=agent_kind != "agentic_data_no_validation",
                memory_enabled=agent_kind != "agentic_data_no_memory",
                optimizer_enabled=agent_kind != "agentic_data_no_optimizer",
                long_term_memory_path=(
                    config.run.output_dir.parent / "agenticdata_memory"
                ),
            ),
        )
    elif agent_kind == "dagent":
        agent = DAgentLiteAgent(
            model=effective_model,
            tools=effective_tools,
            config=DAgentLiteConfig(max_steps=config.agent.max_steps),
        )
    elif agent_kind.startswith("mini_aop"):
        answer_review_enabled = agent_kind not in {"mini_aop_no_review"}
        parallel_prefetch_enabled = agent_kind not in {"mini_aop_no_parallel"}
        agent = MiniAOPAgent(
            model=effective_model,
            tools=effective_tools,
            config=MiniAOPAgentConfig(
                max_steps=config.agent.max_steps,
                answer_review_enabled=answer_review_enabled,
                parallel_prefetch_enabled=parallel_prefetch_enabled,
            ),
        )
    else:
        agent = ReActAgent(
            model=effective_model,
            tools=effective_tools,
            config=ReActAgentConfig(max_steps=config.agent.max_steps),
        )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()

    # Drain the queue while the child is alive. Waiting for process.join() first
    # can deadlock when a large Phase 2 trace fills the multiprocessing pipe.
    deadline = monotonic() + timeout_seconds
    result: dict[str, Any] | None = None
    while result is None:
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        try:
            result = queue.get(timeout=min(0.25, remaining))
        except Empty:
            if not process.is_alive():
                break

    if result is None and process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if result is None:
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    analysis_report_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    for step in reversed(run_result.get("steps", [])):
        if not isinstance(step, dict) or step.get("action") != "__dagent_report_generation__":
            continue
        observation = step.get("observation", {})
        if not isinstance(observation, dict):
            continue
        report_markdown = observation.get("report_markdown")
        if not isinstance(report_markdown, str) or not report_markdown.strip():
            continue
        analysis_report_path = task_output_dir / "analysis_report.md"
        analysis_report_path.write_text(report_markdown, encoding="utf-8")
        break

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        analysis_report_path=analysis_report_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def _load_task_artifacts(task_id: str, run_output_dir: Path) -> TaskRunArtifacts | None:
    task_output_dir = run_output_dir / task_id
    trace_path = task_output_dir / "trace.json"
    if not trace_path.is_file():
        return None

    try:
        run_result = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    prediction_csv_path = task_output_dir / "prediction.csv"
    analysis_report_path = task_output_dir / "analysis_report.md"
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path if prediction_csv_path.is_file() else None,
        analysis_report_path=analysis_report_path if analysis_report_path.is_file() else None,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
    agent_kind: str = "react",
) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None and agent_kind == "react":
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            tools=tools,
            agent_kind=agent_kind,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def _run_single_task_safely(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
    agent_kind: str = "react",
) -> TaskRunArtifacts:
    try:
        return run_single_task(
            task_id=task_id,
            config=config,
            run_output_dir=run_output_dir,
            model=model,
            tools=tools,
            agent_kind=agent_kind,
        )
    except Exception as exc:  # noqa: BLE001
        return _write_task_outputs(
            task_id,
            run_output_dir,
            _failure_run_result_payload(task_id, f"Unhandled task error: {exc}"),
        )


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    agent_kind: str = "react",
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    resume_run_dir: Path | None = None,
    retry_failed: bool = False,
) -> tuple[Path, list[TaskRunArtifacts]]:
    if resume_run_dir is None:
        effective_run_id, run_output_dir = create_run_output_dir(
            config.run.output_dir,
            run_id=config.run.run_id,
        )
    else:
        run_output_dir = resume_run_dir.resolve()
        if not run_output_dir.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_output_dir}")
        effective_run_id = run_output_dir.name

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks(task_ids=task_ids)
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None or agent_kind != "react":
        effective_workers = 1

    selected_task_ids = [task.task_id for task in tasks]
    artifacts_by_task: dict[str, TaskRunArtifacts] = {}
    if resume_run_dir is not None:
        for task_id in selected_task_ids:
            artifact = _load_task_artifacts(task_id, run_output_dir)
            if artifact is not None and (artifact.succeeded or not retry_failed):
                artifacts_by_task[task_id] = artifact

    pending_task_ids = [
        task_id for task_id in selected_task_ids if task_id not in artifacts_by_task
    ]

    if effective_workers == 1:
        use_subprocess_timeout = (
            model is None and tools is None and agent_kind == "react"
        )
        shared_model = None if use_subprocess_timeout else (model or build_model_adapter(config))
        shared_tools = None if use_subprocess_timeout else (
            tools or create_default_tool_registry()
        )
        for task_id in pending_task_ids:
            artifact = _run_single_task_safely(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
                agent_kind=agent_kind,
            )
            artifacts_by_task[task_id] = artifact
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    _run_single_task_safely,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                    agent_kind=agent_kind,
                ): index
                for index, task_id in enumerate(pending_task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(pending_task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            for artifact in indexed_artifacts:
                if artifact is not None:
                    artifacts_by_task[artifact.task_id] = artifact

    task_artifacts = [
        artifacts_by_task[task_id]
        for task_id in selected_task_ids
        if task_id in artifacts_by_task
    ]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
