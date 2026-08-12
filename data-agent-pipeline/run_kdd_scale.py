"""KDD 规模跑驱动：10 任务 × 3 agent（react 对照 + dagent-lite/agenticdata-lite 补测）。

每个 agent 用独立 work_dir（pipeline_runs/kdd_p1_10_<agent>），避免 collect 互相覆盖。
末位打印三 agent 的 summary 横向对比。

用法:
    .venv\\Scripts\\python.exe run_kdd_scale.py --baseline-project C:/Users/86152/Desktop/data-agent-evaluation/baseline
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

AGENTS = ["react", "dagent-lite", "agenticdata-lite"]


def main() -> None:
    parser = argparse.ArgumentParser(description="KDD Phase 1 10-task scale run across 3 agents")
    parser.add_argument("--suite", default="pipeline_smoke_kdd_p1_10.json")
    parser.add_argument("--baseline-project", required=True)
    parser.add_argument("--task-timeout", type=int, default=600)
    args = parser.parse_args()

    from eval_pipeline import run_pipeline

    suite = Path(args.suite)
    if not suite.is_absolute():
        suite = BASE / suite
    bp = Path(args.baseline_project)

    summaries: dict[str, dict] = {}
    for agent in AGENTS:
        work_dir = BASE / "pipeline_runs" / f"kdd_p1_10_{agent}"
        print(f"\n===== agent={agent} work_dir={work_dir} =====", flush=True)
        try:
            run_pipeline(
                suite_path=suite, agent=agent, mock=False, benchmark="kdd",
                baseline_project=bp, work_dir=work_dir, task_timeout=args.task_timeout,
            )
        except Exception as e:  # noqa: BLE001 —— 单个 agent 失败不阻断后续
            print(f"[error] {agent}: {e}", flush=True)
            summaries[agent] = {"error": str(e)}
            continue
        ev = json.loads((work_dir / "evaluation.json").read_text(encoding="utf-8"))
        summaries[agent] = ev["summary"]
        s = ev["summary"]
        print(f"[{agent}] run_success={s['run_success']}/{s['total']} "
              f"correct={s['correct']} wrong={s['wrong_answer']} "
              f"infra_failed={s['infra_failed']} avg_final={s['avg_final_score']}", flush=True)

    # 横向对比
    print("\n========== 三 agent 对比 ==========")
    print(f"{'agent':<16} {'total':>6} {'run_ok':>7} {'correct':>8} {'wrong':>6} {'infra':>6} {'avg':>7}")
    for agent, s in summaries.items():
        if "error" in s:
            print(f"{agent:<16} ERROR: {s['error'][:80]}")
            continue
        print(f"{agent:<16} {s['total']:>6} {s['run_success']:>7} {s['correct']:>8} "
              f"{s['wrong_answer']:>6} {s['infra_failed']:>6} {s['avg_final_score']:>7}")


if __name__ == "__main__":
    main()
