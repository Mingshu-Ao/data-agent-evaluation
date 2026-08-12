"""Level 1 联通测试：Pipeline BaselineAdapter ↔ GitHub Baseline 仓库（真实 dabench subprocess）。

不依赖 crewai（评分 ScoreFlow 另行在全量 eval_pipeline 测）。
验证：prepare_task_dir → run_agent(subprocess) → collect_task_outputs → parse → classify_failure。

用法:
    E:\\Anaconda\\python.exe test_baseline_connect.py --baseline-project C:/Users/86152/Desktop/data-agent-evaluation/baseline
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from baseline_adapter import BaselineAdapter
from loaders.kdd_loader import KDDLoader
from suite_loader import SuiteLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-project", required=True,
                        help="GitHub baseline 仓库根目录")
    parser.add_argument("--suite", default="pipeline_smoke_phase1_easy_3.json")
    parser.add_argument("--task-timeout", type=int, default=600)
    args = parser.parse_args()

    suite = SuiteLoader(BASE / args.suite)
    task_ids = suite.task_ids()
    print(f"suite={suite.suite_name} tasks={task_ids}")

    # 1) prepare_task_dir：把 KDD 原始数据 stage 成统一 task_N/task.json+context、gold.csv
    loader = KDDLoader(input_dir=BASE / "data" / "kdd_phase1" / "input",
                       output_dir=BASE / "data" / "kdd_phase1" / "output")
    dataset_dir = BASE / "pipeline_runs" / "connect_test" / "dataset"
    for tid in task_ids:
        ti, to = BaselineAdapter.prepare_task_dir(loader, tid, dataset_dir)
        assert (ti / "task.json").exists() and (ti / "context").is_dir()
        assert (to / "gold.csv").exists() if loader.load_task(tid).gold_answer else True
        print(f"  [stage] {tid}: task.json + context + gold.csv OK")
    tmp_suite = BASE / "pipeline_runs" / "connect_test" / "tmp_suite.json"
    tmp_suite.write_text(
        json.dumps({"suite_name": suite.suite_name, "task_ids": task_ids}, indent=2),
        encoding="utf-8")

    # 2) run_agent：subprocess 调 dabench（真实模型 API）
    adapter = BaselineAdapter(project_dir=Path(args.baseline_project))
    out_root = BASE / "pipeline_runs" / "connect_test"
    result = adapter.run_agent(
        agent="react", dataset_dir=dataset_dir, suite_path=tmp_suite,
        output_dir=out_root, task_timeout=args.task_timeout,
    )
    print(f"run_agent success={result.get('success')}")
    print(f"  run_id   = {result.get('run_id')}")
    print(f"  run_dir  = {result.get('run_dir')}")
    print(f"  error    = {result.get('error')}")
    print(f"  stderr   = {(result.get('stderr') or '')[-800:]}")
    if not result.get("success"):
        print("[FAIL] dabench 调用失败"); return

    # 3) collect_task_outputs
    run_dir = Path(result["run_dir"])
    collected = BaselineAdapter.collect_task_outputs(run_dir, out_root, task_ids)
    print("collect_task_outputs:", collected)

    # 4) 解析 + 失败分类
    for tid in task_ids:
        pred = adapter.parse_prediction(out_root / tid)
        trace = adapter.parse_trace(out_root / tid)
        failure = adapter.classify_failure(trace, timeout=args.task_timeout)
        print(f"  [{tid}] pred={'YES' if pred else 'no'} "
              f"steps={len(trace.get('steps', [])) if trace else 0} "
              f"succeeded={trace.get('succeeded') if trace else None} "
              f"failure={failure}")
        if trace:
            print(f"        failure_reason={(trace.get('failure_reason') or '')[:160]}")
            ans = trace.get("answer")
            if isinstance(ans, dict):
                print(f"        answer columns={ans.get('columns')} rows={len(ans.get('rows', []))}")

    # 5) summary.json 解析
    summary = BaselineAdapter.parse_summary(run_dir)
    if summary:
        print(f"dabench summary.json: task_count={summary.get('task_count')} "
              f"succeeded={summary.get('succeeded_task_count')}")


if __name__ == "__main__":
    main()
