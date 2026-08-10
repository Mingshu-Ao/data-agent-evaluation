import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.agents.ace_playbook import curate_ace_playbook_from_run
from data_agent_baseline.benchmark.agentic_data_report import (
    write_agentic_data_report,
)
from data_agent_baseline.benchmark.aop_report import write_mini_aop_report
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.dataspace import (
    evaluate_dataspace_run,
    profile_dataspace,
    write_dataspace_matrix_comparison,
    write_dataspace_report_and_suites,
)
from data_agent_baseline.benchmark.dataspace_download import (
    DATASPACE_ARCHIVE_URL,
    DEFAULT_LOCAL_SMOKE_TASK_IDS,
    download_dataspace_subset,
)
from data_agent_baseline.benchmark.evaluation import compare_runs, write_evaluation_outputs
from data_agent_baseline.benchmark.fdabench import (
    materialize_fdabench_multiple_replay,
    write_fdabench_replay_evaluation,
    write_fdabench_report_and_suite,
)
from data_agent_baseline.benchmark.human_review import write_human_review_queue
from data_agent_baseline.benchmark.phase_comparison import (
    write_kdd_phase_comparison,
)
from data_agent_baseline.benchmark.phase_run_comparison import (
    write_phase_run_comparison,
)
from data_agent_baseline.benchmark.reward import write_reward_report
from data_agent_baseline.benchmark.step_limit_report import write_step_limit_report
from data_agent_baseline.benchmark.suites import (
    build_suite_payload,
    load_suite_task_ids,
    write_suite,
)
from data_agent_baseline.benchmark.video_preprocessing import (
    materialize_video_ocr_dataset,
)
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"
DATASPACE_EVALUATOR = (
    PROJECT_ROOT.parents[2] / "dataspace-official" / "evaluation" / "evaluate.py"
)

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _resolve_run_dir(raw_run_dir: Path) -> Path:
    raw_text = str(raw_run_dir)
    if raw_text.lower() == "latest":
        run_dirs = sorted(
            (path for path in ARTIFACT_RUNS_DIR.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            raise typer.BadParameter("No run directories found under artifacts/runs.")
        return run_dirs[0]

    if any(token in raw_text for token in ("你的run_id", "react_run_id", "mini_aop_run_id")):
        raise typer.BadParameter(
            "This looks like a placeholder. Use a real directory name under artifacts/runs, "
            "or use `latest`."
        )

    resolved = raw_run_dir
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_dir():
        raise typer.BadParameter(f"Run directory does not exist: {resolved}")
    return resolved


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_suite_task_ids_option(suite: Path | None) -> list[str] | None:
    if suite is None:
        return None
    return load_suite_task_ids(_resolve_project_path(suite))


def _agentic_data_kind(ablation: str | None) -> str:
    if ablation is None:
        return "agentic_data"
    normalized = ablation.strip().lower().replace("-", "_")
    allowed = {"profile", "edge_profile", "validation", "memory", "optimizer"}
    if normalized not in allowed:
        choices = ", ".join(sorted(value.replace("_", "-") for value in allowed))
        raise typer.BadParameter(
            f"Unknown AgenticData ablation '{ablation}'. Choose one of: {choices}.",
            param_hint="--ablation",
        )
    return f"agentic_data_no_{normalized}"


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row(
        "dataset_root",
        str(app_config.dataset.root_path),
        _status_value(app_config.dataset.root_path),
    )
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-task-mini-aop")
def run_task_mini_aop_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the Mini-AOP plan-first agent on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(
        task_id=task_id,
        config=app_config,
        run_output_dir=run_output_dir,
        agent_kind="mini_aop",
    )

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-task-dagent")
def run_task_dagent_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the paper-inspired DAgent-lite baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir,
            run_id=app_config.run.run_id,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(
        task_id=task_id,
        config=app_config,
        run_output_dir=run_output_dir,
        agent_kind="dagent",
    )

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.analysis_report_path is not None:
        console.print(f"Analysis report: {artifacts.analysis_report_path}")
    else:
        console.print("Analysis report: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-task-agenticdata")
def run_task_agenticdata_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    ablation: str | None = typer.Option(
        None,
        help="Disable one component: profile, edge-profile, validation, memory, or optimizer.",
    ),
) -> None:
    """Run the paper-inspired AgenticData-lite baseline on one task."""
    app_config = load_app_config(config)
    agent_kind = _agentic_data_kind(ablation)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir,
            run_id=app_config.run.run_id,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(
        task_id=task_id,
        config=app_config,
        run_output_dir=run_output_dir,
        agent_kind=agent_kind,
    )

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("evaluate-run")
def evaluate_run_command(
    run_dir: Path = typer.Argument(..., file_okay=False, help="Run output directory, or `latest`."),
    gold_root: Path | None = typer.Option(None, file_okay=False, help="Gold output root."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Evaluate predictions in a run directory against gold.csv files."""
    app_config = load_app_config(config)
    effective_run_dir = _resolve_run_dir(run_dir)
    effective_gold_root = gold_root
    if effective_gold_root is None:
        effective_gold_root = app_config.dataset.root_path.parent / "output"
    evaluation_path, report_path = write_evaluation_outputs(effective_run_dir, effective_gold_root)
    console.print(f"Run directory: {effective_run_dir}")
    console.print(f"Evaluation JSON: {evaluation_path}")
    console.print(f"Error analysis: {report_path}")


@app.command("compare-runs")
def compare_runs_command(
    baseline_run_dir: Path = typer.Argument(
        ..., file_okay=False, help="Baseline run output directory."
    ),
    candidate_run_dir: Path = typer.Argument(
        ..., file_okay=False, help="Candidate run output directory, or `latest`."
    ),
    gold_root: Path | None = typer.Option(None, file_okay=False, help="Gold output root."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Compare two run directories task by task."""
    app_config = load_app_config(config)
    effective_baseline_run_dir = _resolve_run_dir(baseline_run_dir)
    effective_candidate_run_dir = _resolve_run_dir(candidate_run_dir)
    effective_gold_root = gold_root
    if effective_gold_root is None:
        effective_gold_root = app_config.dataset.root_path.parent / "output"
    result = compare_runs(
        effective_baseline_run_dir, effective_candidate_run_dir, effective_gold_root
    )
    output_path = effective_candidate_run_dir / "run_comparison.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    console.print(f"Baseline run: {effective_baseline_run_dir}")
    console.print(f"Candidate run: {effective_candidate_run_dir}")
    console.print(f"Comparison JSON: {output_path}")
    console.print(f"Improved: {result['improved_count']}")
    console.print(f"Regressed: {result['regressed_count']}")
    console.print(f"Unchanged correct: {result['unchanged_correct_count']}")
    console.print(f"Unchanged wrong: {result['unchanged_wrong_count']}")


@app.command("analyze-mini-aop-run")
def analyze_mini_aop_run_command(
    run_dir: Path = typer.Argument(
        ..., file_okay=False, help="Mini-AOP run output directory, or `latest`."
    ),
) -> None:
    """Summarize Mini-AOP planning, DAG rewrite, and answer review traces."""
    effective_run_dir = _resolve_run_dir(run_dir)
    json_path, markdown_path = write_mini_aop_report(effective_run_dir)
    console.print(f"Run directory: {effective_run_dir}")
    console.print(f"Mini-AOP JSON report: {json_path}")
    console.print(f"Mini-AOP Markdown report: {markdown_path}")


@app.command("analyze-agenticdata-run")
def analyze_agenticdata_run_command(
    run_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        help="AgenticData-lite run output directory, or `latest`.",
    ),
) -> None:
    """Analyze AgenticData-lite traces without making model requests."""
    effective_run_dir = _resolve_run_dir(run_dir)
    json_path, markdown_path = write_agentic_data_report(effective_run_dir)
    console.print(f"Run directory: {effective_run_dir}")
    console.print(f"AgenticData JSON report: {json_path}")
    console.print(f"AgenticData Markdown report: {markdown_path}")


@app.command("reward-run")
def reward_run_command(
    run_dir: Path = typer.Argument(..., file_okay=False, help="Run output directory, or `latest`."),
    gold_root: Path | None = typer.Option(None, file_okay=False, help="Gold output root."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Score a run with a heuristic reward model for offline feedback."""
    app_config = load_app_config(config)
    effective_run_dir = _resolve_run_dir(run_dir)
    effective_gold_root = gold_root or app_config.dataset.root_path.parent / "output"
    json_path, markdown_path = write_reward_report(effective_run_dir, effective_gold_root)
    console.print(f"Reward JSON report: {json_path}")
    console.print(f"Reward Markdown report: {markdown_path}")


@app.command("create-human-review")
def create_human_review_command(
    run_dir: Path = typer.Argument(..., file_okay=False, help="Run output directory, or `latest`."),
    gold_root: Path | None = typer.Option(None, file_okay=False, help="Gold output root."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Create a human-in-the-loop review queue for failed or partially correct tasks."""
    app_config = load_app_config(config)
    effective_run_dir = _resolve_run_dir(run_dir)
    effective_gold_root = gold_root or app_config.dataset.root_path.parent / "output"
    json_path, markdown_path = write_human_review_queue(
        run_dir=effective_run_dir,
        gold_root=effective_gold_root,
        dataset_root=app_config.dataset.root_path,
    )
    console.print(f"Human review JSON: {json_path}")
    console.print(f"Human review Markdown: {markdown_path}")


@app.command("create-coverage-suite")
def create_coverage_suite_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    output: Path = typer.Option(
        Path("configs/suites/phase1_coverage_20.json"), help="Output suite JSON path."
    ),
    size: int = typer.Option(20, min=1, help="Number of tasks to include."),
    name: str = typer.Option("phase1_coverage_20", help="Suite name."),
    require_file_types: str = typer.Option(
        "",
        help="Comma-separated file types every selected task must contain.",
    ),
    exclude_file_types: str = typer.Option(
        "",
        help="Comma-separated file types to exclude, for example mp4.",
    ),
) -> None:
    """Create a coverage-oriented task suite JSON file."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    payload = build_suite_payload(
        dataset=dataset,
        suite_name=name,
        suite_size=size,
        description=(
            "Coverage-oriented public-task suite covering difficulties, "
            "context file combinations, and question types."
        ),
        require_file_types={
            value.strip() for value in require_file_types.split(",") if value.strip()
        },
        exclude_file_types={
            value.strip() for value in exclude_file_types.split(",") if value.strip()
        },
    )
    output_path = _resolve_project_path(output)
    write_suite(output_path, payload)
    console.print(f"Suite JSON: {output_path}")
    console.print(f"Tasks: {payload['suite_summary']['task_count']}")
    console.print(f"Difficulties: {payload['suite_summary']['difficulty_counts']}")
    console.print(f"File type sets: {payload['suite_summary']['file_type_set_counts']}")
    console.print(f"Question types: {payload['suite_summary']['question_type_counts']}")


@app.command("compare-kdd-phases")
def compare_kdd_phases_command(
    phase1_input: Path = typer.Option(..., exists=True, file_okay=False),
    phase1_gold: Path = typer.Option(..., exists=True, file_okay=False),
    phase2_input: Path = typer.Option(..., exists=True, file_okay=False),
    phase2_gold: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("reports/dataset_comparison")),
) -> None:
    """Compare Phase 1 and Phase 2 task, modality, and gold-output profiles."""
    json_path, markdown_path = write_kdd_phase_comparison(
        phase1_input=_resolve_project_path(phase1_input),
        phase1_gold=_resolve_project_path(phase1_gold),
        phase2_input=_resolve_project_path(phase2_input),
        phase2_gold=_resolve_project_path(phase2_gold),
        output_dir=_resolve_project_path(output_dir),
    )
    console.print(f"Comparison JSON: {json_path}")
    console.print(f"Comparison Markdown: {markdown_path}")


@app.command("compare-phase-runs")
def compare_phase_runs_command(
    phase1_run: Path = typer.Option(..., exists=True, file_okay=False),
    phase1_gold: Path = typer.Option(..., exists=True, file_okay=False),
    phase2_run: Path = typer.Option(..., exists=True, file_okay=False),
    phase2_gold: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("reports/phase_run_comparison")),
    phase1_label: str = typer.Option("KDD Phase 1"),
    phase2_label: str = typer.Option("KDD Phase 2"),
) -> None:
    """Compare success and accuracy rates from independent Phase 1/2 runs."""
    json_path, markdown_path = write_phase_run_comparison(
        phase1_run_dir=_resolve_project_path(phase1_run),
        phase1_gold_root=_resolve_project_path(phase1_gold),
        phase2_run_dir=_resolve_project_path(phase2_run),
        phase2_gold_root=_resolve_project_path(phase2_gold),
        output_dir=_resolve_project_path(output_dir),
        phase1_label=phase1_label,
        phase2_label=phase2_label,
    )
    console.print(f"Comparison JSON: {json_path}")
    console.print(f"Comparison Markdown: {markdown_path}")


@app.command("diagnose-step-limit")
def diagnose_step_limit_command(
    run_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        help="Run output directory, or `latest`.",
    ),
) -> None:
    """Explain why tasks exhausted their configured agent step limit."""
    effective_run_dir = _resolve_run_dir(run_dir)
    json_path, markdown_path = write_step_limit_report(effective_run_dir)
    console.print(f"Step-limit JSON: {json_path}")
    console.print(f"Step-limit Markdown: {markdown_path}")


@app.command("download-dataspace-subset")
def download_dataspace_subset_command(
    output_root: Path = typer.Option(
        Path("data/dataspace_local_subset"),
        help="Local directory for the partial DataSpace extraction.",
    ),
    task_ids: str = typer.Option(
        ",".join(DEFAULT_LOCAL_SMOKE_TASK_IDS),
        help="Comma-separated public-reference task ids.",
    ),
    archive_url: str = typer.Option(
        DATASPACE_ARCHIVE_URL,
        help="Official DataSpace ZIP URL.",
    ),
) -> None:
    """Download selected DataSpace tasks with HTTP Range requests."""
    selected_ids = [value.strip() for value in task_ids.split(",") if value.strip()]

    def on_file(index: int, total: int, member_name: str, reused: bool) -> None:
        state = "reused" if reused else "downloaded"
        console.print(f"[{index}/{total}] {state}: {member_name}")

    try:
        manifest = download_dataspace_subset(
            output_root=_resolve_project_path(output_root),
            task_ids=selected_ids,
            archive_url=archive_url,
            progress_callback=on_file,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Benchmark root: {manifest['benchmark_root']}")
    console.print(f"Tasks downloaded: {manifest['task_count']}")
    console.print(
        "Uncompressed bytes downloaded this run: "
        f"{manifest['downloaded_uncompressed_bytes']}"
    )
    console.print(f"Manifest: {manifest['manifest_path']}")


@app.command("prepare-dataspace")
def prepare_dataspace_command(
    benchmark_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Extracted DataSpace-Benchmark directory or its parent.",
    ),
    report_dir: Path = typer.Option(
        Path("artifacts/dataspace"),
        help="Directory for DataSpace inventory reports.",
    ),
    suites_dir: Path = typer.Option(
        Path("configs/suites"),
        help="Directory for generated public-reference suites.",
    ),
    smoke_size: int = typer.Option(5, min=1, help="Smoke suite size."),
    coverage_size: int = typer.Option(20, min=1, help="Coverage suite size."),
) -> None:
    """Scan DataSpace and generate deterministic public-reference suites."""
    try:
        paths = write_dataspace_report_and_suites(
            benchmark_root=benchmark_root,
            report_dir=_resolve_project_path(report_dir),
            suites_dir=_resolve_project_path(suites_dir),
            smoke_size=smoke_size,
            coverage_size=coverage_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--benchmark-root") from exc
    for label, path in paths.items():
        console.print(f"{label}: {path}")


@app.command("evaluate-dataspace")
def evaluate_dataspace_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    benchmark_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Extracted DataSpace-Benchmark directory or its parent.",
    ),
    evaluator_script: Path = typer.Option(
        DATASPACE_EVALUATOR,
        exists=True,
        dir_okay=False,
        help="Official DataSpace evaluation/evaluate.py path.",
    ),
    suite: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional suite JSON; omitted means all 60 public references.",
    ),
) -> None:
    """Run the official evaluator and add non-exclusive modality slices."""
    task_ids = _load_suite_task_ids_option(suite)
    try:
        paths = evaluate_dataspace_run(
            run_dir=run_dir.resolve(),
            benchmark_root=benchmark_root,
            evaluator_script=evaluator_script,
            task_ids=task_ids,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for label, path in paths.items():
        console.print(f"{label}: {path}")


@app.command("run-dataspace-matrix")
def run_dataspace_matrix_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    benchmark_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Extracted DataSpace-Benchmark directory or its parent.",
    ),
    evaluator_script: Path = typer.Option(
        DATASPACE_EVALUATOR,
        exists=True,
        dir_okay=False,
        help="Official DataSpace evaluation/evaluate.py path.",
    ),
    suite: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Suite JSON; omitted means all public-reference tasks.",
    ),
    limit: int | None = typer.Option(None, min=1, help="Maximum tasks per agent."),
    agents: str = typer.Option(
        "react,dagent-lite,agenticdata-lite,mini-aop",
        help="Comma-separated agents to run.",
    ),
) -> None:
    """Run and officially score a same-model DataSpace baseline matrix."""
    aliases = {
        "react": ("react", "react"),
        "react-enhanced": ("react-enhanced", "react_enhanced"),
        "react-enhanced-no-schema": (
            "react-enhanced-no-schema",
            "react_enhanced_no_schema",
        ),
        "react-enhanced-no-convergence": (
            "react-enhanced-no-convergence",
            "react_enhanced_no_convergence",
        ),
        "react-enhanced-no-ace": (
            "react-enhanced-no-ace",
            "react_enhanced_no_ace",
        ),
        "dagent": ("dagent-lite", "dagent"),
        "dagent-lite": ("dagent-lite", "dagent"),
        "agenticdata": ("agenticdata-lite", "agentic_data"),
        "agenticdata-lite": ("agenticdata-lite", "agentic_data"),
        "mini-aop": ("mini-aop", "mini_aop"),
        "mini_aop": ("mini-aop", "mini_aop"),
    }
    requested = [value.strip().lower() for value in agents.split(",") if value.strip()]
    unknown = [value for value in requested if value not in aliases]
    if not requested or unknown:
        allowed = (
            "react, dagent-lite, agenticdata-lite, mini-aop, react-enhanced, "
            "react-enhanced-no-schema, react-enhanced-no-convergence, "
            "react-enhanced-no-ace"
        )
        detail = f" Unknown: {', '.join(unknown)}." if unknown else ""
        raise typer.BadParameter(f"Choose one or more of: {allowed}.{detail}", param_hint="--agents")

    selected_agents: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for value in requested:
        label, kind = aliases[value]
        if label not in seen_labels:
            selected_agents.append((label, kind))
            seen_labels.add(label)

    app_config = load_app_config(config)
    suite_task_ids = _load_suite_task_ids_option(suite)
    if suite_task_ids is None:
        _, profiles = profile_dataspace(benchmark_root)
        suite_task_ids = [profile.task_id for profile in profiles if profile.public_reference]
    suite_task_ids = sorted(
        suite_task_ids,
        key=lambda task_id: int(task_id.removeprefix("task_")),
    )
    if limit is not None:
        suite_task_ids = suite_task_ids[:limit]
    if not suite_task_ids:
        raise typer.BadParameter("No tasks selected for the matrix.", param_hint="--suite")

    available_task_ids = set(DABenchPublicDataset(app_config.dataset.root_path).list_task_ids())
    missing = [task_id for task_id in suite_task_ids if task_id not in available_task_ids]
    if missing:
        raise typer.BadParameter(
            "The configured dataset.root_path is missing selected tasks: "
            f"{', '.join(missing[:10])}",
            param_hint="--config",
        )

    matrix_id = datetime.now(timezone.utc).strftime("dataspace_matrix_%Y%m%dT%H%M%S%fZ")
    matrix_dir = app_config.run.output_dir / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=False)
    agent_runs: dict[str, Path] = {}
    manifest_agents: list[dict[str, object]] = []

    console.print(
        f"DataSpace matrix: {len(suite_task_ids)} tasks x {len(selected_agents)} agents"
    )
    for label, agent_kind in selected_agents:
        console.print(f"Starting {label}...")
        completed_count = 0

        def on_task_complete(artifact: TaskRunArtifacts) -> None:
            nonlocal completed_count
            completed_count += 1
            console.print(
                f"  [{completed_count}/{len(suite_task_ids)}] {artifact.task_id}: "
                f"{'ok' if artifact.succeeded else 'fail'}"
            )

        fair_run_config = replace(
            app_config,
            run=replace(
                app_config.run,
                output_dir=matrix_dir,
                run_id=label,
                max_workers=1,
            ),
        )
        try:
            run_dir, artifacts = run_benchmark(
                config=fair_run_config,
                task_ids=suite_task_ids,
                agent_kind=agent_kind,
                progress_callback=on_task_complete,
            )
            evaluation_paths = evaluate_dataspace_run(
                run_dir=run_dir,
                benchmark_root=benchmark_root,
                evaluator_script=evaluator_script,
                task_ids=suite_task_ids,
            )
        except (FileNotFoundError, RuntimeError, ValueError, FileExistsError) as exc:
            raise typer.BadParameter(f"{label} failed: {exc}") from exc
        agent_runs[label] = run_dir
        manifest_agents.append(
            {
                "agent": label,
                "agent_kind": agent_kind,
                "run_dir": str(run_dir),
                "attempted": len(artifacts),
                "submitted": sum(artifact.succeeded for artifact in artifacts),
                "evaluation": str(evaluation_paths["evaluation"]),
            }
        )

    manifest_path = matrix_dir / "matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "matrix_id": matrix_id,
                "model": app_config.agent.model,
                "api_base": app_config.agent.api_base,
                "temperature": app_config.agent.temperature,
                "max_steps": app_config.agent.max_steps,
                "task_timeout_seconds": app_config.run.task_timeout_seconds,
                "max_workers": 1,
                "task_ids": suite_task_ids,
                "agents": manifest_agents,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_paths = write_dataspace_matrix_comparison(
        matrix_dir=matrix_dir,
        agent_runs=agent_runs,
    )
    console.print(f"Matrix output: {matrix_dir}")
    console.print(f"Manifest: {manifest_path}")
    for label, path in comparison_paths.items():
        console.print(f"{label}: {path}")


@app.command("curate-ace-playbook")
def curate_ace_playbook_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    evaluation: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Official evaluation JSON; defaults to RUN_DIR/dataspace_evaluation.json.",
    ),
    playbook: Path | None = typer.Option(
        None,
        dir_okay=False,
        help="ACE-lite playbook path; defaults beside the configured run output directory.",
    ),
) -> None:
    """Reflect an adaptation run into a persistent, answer-free ACE-lite playbook."""
    app_config = load_app_config(config)
    evaluation_path = evaluation or run_dir / "dataspace_evaluation.json"
    playbook_path = (
        playbook
        or app_config.agent.ace_playbook_path
        or app_config.run.output_dir.parent / "ace_playbook.json"
    )
    if not evaluation_path.is_file():
        raise typer.BadParameter(
            f"Official evaluation JSON does not exist: {evaluation_path}",
            param_hint="--evaluation",
        )
    report = curate_ace_playbook_from_run(
        run_dir=run_dir.resolve(),
        dataset_root=app_config.dataset.root_path,
        evaluation_path=evaluation_path.resolve(),
        playbook_path=_resolve_project_path(playbook_path),
    )
    console.print(f"Playbook: {report['playbook_path']}")
    console.print(f"Deltas curated: {report['delta_count']}")
    console.print(f"Playbook entries: {report['entry_count']}")


@app.command("prepare-fdabench")
def prepare_fdabench_command(
    root: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("reports/fdabench")),
    suite_size: int = typer.Option(12, min=1),
) -> None:
    """Validate FDABench JSONL splits and create a coverage-oriented suite."""
    report_path, markdown_path, suite_path = write_fdabench_report_and_suite(
        root=_resolve_project_path(root),
        output_dir=_resolve_project_path(output_dir),
        suite_size=suite_size,
    )
    console.print(f"FDABench JSON report: {report_path}")
    console.print(f"FDABench Markdown report: {markdown_path}")
    console.print(f"FDABench suite: {suite_path}")


@app.command("materialize-fdabench-replay")
def materialize_fdabench_replay_command(
    root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path = typer.Option(Path("data/fdabench_replay")),
    size: int = typer.Option(12, min=1),
) -> None:
    """Create a KDD-compatible replay set from released FDABench evidence."""
    input_root, gold_root, manifest_path = materialize_fdabench_multiple_replay(
        root=_resolve_project_path(root),
        output_root=_resolve_project_path(output_root),
        size=size,
    )
    console.print(f"Replay input: {input_root}")
    console.print(f"Replay gold: {gold_root}")
    console.print(f"Replay manifest: {manifest_path}")


@app.command("evaluate-fdabench-replay")
def evaluate_fdabench_replay_command(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    gold_root: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    """Evaluate FDABench multiple-select replay with option-level metrics."""
    json_path, markdown_path = write_fdabench_replay_evaluation(
        run_dir=_resolve_project_path(run_dir),
        gold_root=_resolve_project_path(gold_root),
    )
    console.print(f"FDABench replay evaluation: {json_path}")
    console.print(f"FDABench replay report: {markdown_path}")


@app.command("prepare-video-dataset")
def prepare_video_dataset_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    suite: Path = typer.Option(..., exists=True, dir_okay=False),
    gold_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path = typer.Option(...),
    interval_seconds: float = typer.Option(10.0, min=1.0),
    max_frames: int = typer.Option(12, min=1),
    minimum_confidence: float = typer.Option(0.45, min=0.0, max=1.0),
    evidence_mode: str = typer.Option(
        "ocr",
        help="Evidence variant: ocr, ocr_keyframes, or ocr_keyframes_asr.",
    ),
    scene_probe_interval_seconds: float = typer.Option(1.0, min=0.1),
    scene_change_threshold: float = typer.Option(0.12, min=0.0, max=1.0),
    scene_min_gap_seconds: float = typer.Option(2.0, min=0.0),
    asr_model_name: str = typer.Option("small"),
    asr_language: str | None = typer.Option(
        None,
        help="ISO language code such as zh or en; omit for automatic detection.",
    ),
    asr_device: str = typer.Option("cpu"),
    asr_compute_type: str = typer.Option("int8"),
    attach_keyframes: bool = typer.Option(
        True,
        "--attach-keyframes/--no-attach-keyframes",
        help="Attach selected frame images to model requests; disable for text-only models.",
    ),
    reuse_input_root: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Reuse already processed task directories from this derived input root.",
    ),
) -> None:
    """Create a Phase 2 dataset from OCR, visual keyframes, and optional ASR."""
    app_config = load_app_config(config)
    input_root, output_gold_root, manifest_path = materialize_video_ocr_dataset(
        dataset_root=app_config.dataset.root_path,
        gold_root=_resolve_project_path(gold_root),
        suite_path=_resolve_project_path(suite),
        output_root=_resolve_project_path(output_root),
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        minimum_confidence=minimum_confidence,
        evidence_mode=evidence_mode,
        scene_probe_interval_seconds=scene_probe_interval_seconds,
        scene_change_threshold=scene_change_threshold,
        scene_min_gap_seconds=scene_min_gap_seconds,
        asr_model_name=asr_model_name,
        asr_language=asr_language,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        attach_keyframes=attach_keyframes,
        reuse_input_root=(
            _resolve_project_path(reuse_input_root) if reuse_input_root is not None else None
        ),
        progress_callback=lambda current, total, task_id, reused: console.print(
            f"[{current}/{total}] {task_id}: {'reused' if reused else 'processed'}"
        ),
    )
    console.print(f"Derived input: {input_root}")
    console.print(f"Derived gold: {output_gold_root}")
    console.print(f"Manifest: {manifest_path}")


@app.command("show-suite")
def show_suite_command(
    suite: Path = typer.Argument(..., exists=True, dir_okay=False, help="Suite JSON path."),
) -> None:
    """Show a task suite summary."""
    payload = json.loads(_resolve_project_path(suite).read_text(encoding="utf-8"))
    console.print(f"Suite: {payload.get('suite_name')}")
    console.print(f"Description: {payload.get('description')}")
    console.print(f"Task count: {payload.get('suite_summary', {}).get('task_count')}")
    console.print(f"Difficulties: {payload.get('suite_summary', {}).get('difficulty_counts')}")
    console.print(f"File type sets: {payload.get('suite_summary', {}).get('file_type_set_counts')}")
    console.print(f"Question types: {payload.get('suite_summary', {}).get('question_type_counts')}")
    table = Table(title="Suite Tasks")
    table.add_column("Task")
    table.add_column("Difficulty")
    table.add_column("File Types")
    table.add_column("Question Type")
    for item in payload.get("tasks", []):
        table.add_row(
            str(item.get("task_id")),
            str(item.get("difficulty")),
            "+".join(item.get("file_types", [])),
            str(item.get("question_type")),
        )
    console.print(table)


@app.command("list-runs")
def list_runs_command() -> None:
    """List local run directories under artifacts/runs."""
    table = Table(title="Local Runs")
    table.add_column("Run ID")
    table.add_column("Modified")
    table.add_column("Tasks")
    run_dirs = sorted(
        (path for path in ARTIFACT_RUNS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        task_count = sum(
            1 for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_")
        )
        modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(run_dir.name, modified, str(task_count))
    console.print(table)


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    suite: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Task suite JSON path."
    ),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    suite_task_ids = _load_suite_task_ids_option(suite)
    task_total = len(dataset.iter_tasks(task_ids=suite_task_ids))
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                task_ids=suite_task_ids,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("run-benchmark-mini-aop")
def run_benchmark_mini_aop_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    suite: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Task suite JSON path."
    ),
) -> None:
    """Run the Mini-AOP plan-first agent on multiple tasks."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    suite_task_ids = _load_suite_task_ids_option(suite)
    task_total = len(dataset.iter_tasks(task_ids=suite_task_ids))
    if limit is not None:
        task_total = min(task_total, limit)

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Mini-AOP Benchmark",
            total=task_total,
            completed=0,
            ok="0",
            fail="0",
            speed="rate=0.0 task/min",
            last="last=-",
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Mini-AOP Benchmark",
                refresh=True,
                ok=str(succeeded_count),
                fail=str(failed_count),
                speed=_format_compact_rate(completion_count, perf_counter() - start_time),
                last=_format_last_task(artifact),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                task_ids=suite_task_ids,
                agent_kind="mini_aop",
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("run-benchmark-dagent")
def run_benchmark_dagent_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    suite: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Task suite JSON path.",
    ),
    resume: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Existing run directory to resume; completed tasks are skipped.",
    ),
    retry_failed: bool = typer.Option(
        False,
        help="With --resume, rerun tasks whose saved trace is unsuccessful.",
    ),
) -> None:
    """Run the paper-inspired DAgent-lite baseline on multiple tasks."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    suite_task_ids = _load_suite_task_ids_option(suite)
    task_total = len(dataset.iter_tasks(task_ids=suite_task_ids))
    if limit is not None:
        task_total = min(task_total, limit)
    effective_resume_dir = _resolve_run_dir(resume) if resume is not None else None
    if effective_resume_dir is not None:
        selected_ids = [
            task.task_id for task in dataset.iter_tasks(task_ids=suite_task_ids)[:task_total]
        ]
        completed_ids: set[str] = set()
        for task_id in selected_ids:
            trace_path = effective_resume_dir / task_id / "trace.json"
            if not trace_path.is_file():
                continue
            if not retry_failed:
                completed_ids.add(task_id)
                continue
            try:
                saved_result = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if saved_result.get("succeeded"):
                completed_ids.add(task_id)
        task_total -= len(completed_ids)

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "DAgent-lite Benchmark",
            total=task_total,
            completed=0,
            ok="0",
            fail="0",
            speed="rate=0.0 task/min",
            last="last=-",
        )
        completed_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact: TaskRunArtifacts) -> None:
            nonlocal completed_count, succeeded_count, failed_count
            completed_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completed_count,
                refresh=True,
                ok=str(succeeded_count),
                fail=str(failed_count),
                speed=_format_compact_rate(completed_count, perf_counter() - start_time),
                last=_format_last_task(artifact),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                task_ids=suite_task_ids,
                agent_kind="dagent",
                progress_callback=on_task_complete,
                resume_run_dir=effective_resume_dir,
                retry_failed=retry_failed,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("run-benchmark-agenticdata")
def run_benchmark_agenticdata_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    suite: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Task suite JSON path.",
    ),
    resume: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Existing run directory to resume; completed tasks are skipped.",
    ),
    retry_failed: bool = typer.Option(
        False,
        help="With --resume, rerun tasks whose saved trace is unsuccessful.",
    ),
    ablation: str | None = typer.Option(
        None,
        help="Disable one component: profile, edge-profile, validation, memory, or optimizer.",
    ),
) -> None:
    """Run the paper-inspired AgenticData-lite baseline on multiple tasks."""
    app_config = load_app_config(config)
    agent_kind = _agentic_data_kind(ablation)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    suite_task_ids = _load_suite_task_ids_option(suite)
    selected_tasks = dataset.iter_tasks(task_ids=suite_task_ids)
    if limit is not None:
        selected_tasks = selected_tasks[:limit]
    effective_resume_dir = _resolve_run_dir(resume) if resume is not None else None

    completed_ids: set[str] = set()
    if effective_resume_dir is not None:
        for task in selected_tasks:
            trace_path = effective_resume_dir / task.task_id / "trace.json"
            if not trace_path.is_file():
                continue
            if not retry_failed:
                completed_ids.add(task.task_id)
                continue
            try:
                saved_result = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if saved_result.get("succeeded"):
                completed_ids.add(task.task_id)

    pending_total = len(selected_tasks) - len(completed_ids)
    completed_count = 0
    succeeded_count = 0

    def on_task_complete(artifact: TaskRunArtifacts) -> None:
        nonlocal completed_count, succeeded_count
        completed_count += 1
        if artifact.succeeded:
            succeeded_count += 1
        console.print(
            f"[{completed_count}/{pending_total}] {artifact.task_id}: "
            f"{'ok' if artifact.succeeded else 'fail'}"
        )

    try:
        run_output_dir, artifacts = run_benchmark(
            config=app_config,
            limit=limit,
            task_ids=suite_task_ids,
            agent_kind=agent_kind,
            progress_callback=on_task_complete,
            resume_run_dir=effective_resume_dir,
            retry_failed=retry_failed,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


def main() -> None:
    app()
