"""统一 Data Agent 评测 Pipeline

流程：
  1. 加载 suite JSON → 获取任务列表
  2. KDDLoader 加载原始数据 → 转成统一任务目录（input/task_id/ + output/task_id/）
  3. 调用 Baseline Agent（真实模式：subprocess 调 dabench；mock 模式：自带 Worker）
  4. 解析输出 → 对比 gold → 评分 → 生成统一结果表

用法:
    python eval_pipeline.py --suite pipeline_smoke_phase1_easy_3.json --agent react --mock
    python eval_pipeline.py --suite ... --agent react --baseline-project D:\\bupt\\...\\PHASE_1
"""
from __future__ import annotations
import sys
import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

# GBK 控制台（Windows 默认代码页）打印 emoji/中文会崩，且 crewai 的 trace 也会因此刷错误
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def _ensure_tas_importable(project_dir: Optional[Path] = None) -> None:
    """让 `TAS.Crewai.utils` 可导入（workers 依赖 Baseline 项目的 TAS 包）。

    TAS 不在 Pipeline 的 venv 里，需要把其父目录加入 sys.path。
    候选位置：环境变量 TAS_PATH → --baseline-project/TAS → 常见兄弟目录 → Pipeline 内 TAS。
    """
    try:
        import importlib.util  # noqa: PLC0415
        if importlib.util.find_spec("TAS") is not None:
            return
    except Exception:
        pass

    candidates = []
    env = __import__("os").environ.get("TAS_PATH")
    if env:
        candidates.append(Path(env))
    if project_dir is not None:
        candidates.append(Path(project_dir) / "TAS")
    candidates += [
        BASE.parent / "data_agent" / "miaohongfan",   # 本机实际位置（Desktop/data_agent/miaohongfan/TAS）
        BASE.parent / "miaohongfan",
        BASE / "TAS",
    ]
    for parent in candidates:
        if (parent / "TAS").is_dir():
            p = str(parent)
            # 追加到末尾（低优先级）：TAS 父目录里可能也有 workers/、loaders/ 等
            # 同名镜像包，绝不能放到 Pipeline 目录前面抢走 import。
            if p not in sys.path:
                sys.path.append(p)
            # 强制 Pipeline 自身目录回到最前，保证 workers/loaders 永远用本仓库的。
            base = str(BASE)
            if base in sys.path:
                sys.path.remove(base)
            sys.path.insert(0, base)
            return
    # 找不到时给出明确提示，而不是让 import 抛出晦涩错误
    print(
        "[warn] 未找到 TAS 包（workers 依赖 TAS.Crewai.utils）。"
        "请设置环境变量 TAS_PATH=<TAS 父目录>，或确认 Baseline 项目路径。"
    )


def _load_knowledge_base() -> str:
    """加载 eval_knowledge.json 评测知识，注入 mock 模式的 Worker。"""
    kb_path = BASE / "eval_knowledge.json"
    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    parts = []
    for key, value in kb.items():
        if isinstance(value, dict) and "text" in value:
            parts.append(f"- {key}: {value['text']}")
        elif isinstance(value, dict) and "task" in value:
            parts.append(f"- {key}: {value['task']}")
    return "\n".join(parts)


def _make_loader(benchmark: str, kdd_input: Optional[Path], kdd_output: Optional[Path]):
    """按 benchmark 构造对应的数据 loader（统一 list_tasks/load_task/task_count 接口）。"""
    if benchmark == "fdabench":
        from loaders.fdabench_loader import FDAbenchLoader
        return FDAbenchLoader(cache_dir=str(BASE / "data" / "fdabench"),
                              context_root=str(BASE / "data" / "fdabench_contexts"))
    if benchmark == "krama":
        from loaders.krama_loader import KramaLoader
        return KramaLoader(data_dir=str(BASE / "data" / "kramabench"))
    if benchmark == "lakeqa":
        from loaders.lakeqa_loader import LakeQALoader
        return LakeQALoader(data_dir=str(BASE / "data" / "lakeqa"))
    # 默认 kdd
    from loaders.kdd_loader import KDDLoader
    kdd_input = kdd_input or (BASE / "data" / "kdd_phase1" / "input")
    kdd_output = kdd_output or (BASE / "data" / "kdd_phase1" / "output")
    return KDDLoader(input_dir=kdd_input, output_dir=kdd_output)


def run_pipeline(
    suite_path: Path,
    agent: str = "react",
    mock: bool = False,
    benchmark: str = "kdd",
    kdd_input: Optional[Path] = None,
    kdd_output: Optional[Path] = None,
    baseline_project: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    venv_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    task_timeout: int = 600,
):
    """完整 Pipeline 流程"""
    # 先保证 TAS 可导入（workers 依赖），再 import 任何 worker
    _ensure_tas_importable(project_dir=baseline_project)
    from suite_loader import SuiteLoader
    from baseline_adapter import BaselineAdapter, TaskIdMapper
    from workers.score_worker import ScoreFlow

    suite = SuiteLoader(suite_path)
    task_ids = suite.task_ids()
    if not task_ids:
        print(f"[error] suite 没有任务: {suite_path}")
        return []
    work_dir = work_dir or BASE / "pipeline_runs" / suite.suite_name
    work_dir.mkdir(parents=True, exist_ok=True)

    mode = "MOCK" if mock else f"Baseline/{agent}"
    print(f"=== Pipeline: {suite.suite_name} ({len(task_ids)} tasks, benchmark={benchmark}, mode={mode}) ===")

    loader = _make_loader(benchmark, kdd_input, kdd_output)
    # 大数据湖 benchmark（Krama/LakeQA）用 symlink 而不是复制整个数据目录
    use_symlink_context = benchmark in ("krama", "lakeqa")
    # 单 cell 自由文本/数值答案的 benchmark → LLM 语义评分（KDD 表格仍走 ScoreFlow 行相等）
    semantic_scoring = benchmark in ("fdabench", "krama", "lakeqa")

    if mock:
        from workers.structured_query import StructuredQueryFlow
        from workers.python_analysis import PythonAnalysisFlow
        from workers.answer_worker import AnswerFlow

        knowledge_text = _load_knowledge_base()
        results = []
        scores: dict[str, float] = {}
        for tid in task_ids:
            task = loader.load_task(tid)
            start = time.time()
            trace, pred = _run_mock_task(
                loader=loader, tid=tid, task=task,
                knowledge_text=knowledge_text,
            )
            elapsed = time.time() - start

            # 写统一输出
            task_dir = work_dir / tid
            task_dir.mkdir(parents=True, exist_ok=True)
            if pred is not None:
                with (task_dir / "prediction.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for row in pred:
                        w.writerow(row)
            trace.setdefault("e2e_elapsed_seconds", round(elapsed, 3))
            with (task_dir / "trace.json").open("w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

            score = _score_task(task, pred, semantic=semantic_scoring)
            results.append(_build_record(
                benchmark=benchmark, agent="mock", task_id=tid,
                pred=pred, trace=trace, score=score,
                task_dir=task_dir, config_id="mock", code_version="mock",
                failure_hint=None,
            ))
            scores[tid] = float(score.get("final_score", 0))
            _print_progress(tid, score, pred)
    else:
        # 真实 Baseline 模式
        # ID 映射：非 task_<int> 的 benchmark ID（FDA0001/legal-hard-1/lakeqa-full:EQA...）
        # 必须映射成 task_<int> 才能过 Baseline 的 _task_number 校验；对外 records 仍用原始 ID。
        mapper = TaskIdMapper(task_ids)
        baseline_ids = [mapper.to_baseline(t) for t in task_ids]
        id_map = {mapper.to_baseline(t): t for t in task_ids}

        dataset_dir = work_dir / "dataset"
        print(f"Preparing task dirs in {dataset_dir}...")
        for tid, btid in zip(task_ids, baseline_ids):
            BaselineAdapter.prepare_task_dir(loader, tid, dataset_dir,
                                             use_symlink=use_symlink_context, baseline_id=btid)
        tmp_suite = work_dir / "tmp_suite.json"
        tmp_suite.write_text(json.dumps({"suite_name": suite.suite_name, "task_ids": baseline_ids}, indent=2))

        adapter = BaselineAdapter(
            project_dir=baseline_project or Path("."),
            venv_dir=venv_dir,
            config_path=config_path,
        )
        result = adapter.run_agent(
            agent=agent, dataset_dir=dataset_dir, suite_path=tmp_suite,
            output_dir=work_dir, task_timeout=task_timeout,
        )
        if not result.get("success"):
            print(f"[warn] Baseline 调用未完全成功: {result.get('error', 'unknown')}")
        run_model = result.get("model") or "deepseek-chat"
        run_dir = Path(result.get("run_dir") or "")
        if run_dir.exists():
            BaselineAdapter.collect_task_outputs(run_dir, work_dir, baseline_ids, id_map=id_map)
        else:
            print(f"[warn] Baseline run 目录不存在: {run_dir}")

        results = []
        scores: dict[str, float] = {}
        for tid in task_ids:
            task = loader.load_task(tid)
            pred = adapter.parse_prediction(work_dir / tid)
            trace = adapter.parse_trace(work_dir / tid)
            score = _score_task(task, pred, semantic=semantic_scoring)
            # 失败分类：无预测 → 按 trace 归类；有预测但答错 → wrong_answer；答对 → ""
            failure = adapter.classify_failure(trace, timeout=task_timeout) if not pred else "wrong_answer"
            if pred and score.get("final_score", 0) >= 0.99:
                failure = ""
            results.append(_build_record(
                benchmark=benchmark, agent=agent, task_id=tid,
                pred=pred, trace=trace, score=score,
                task_dir=work_dir / tid, config_id=agent, code_version="baseline",
                failure_hint=failure, model=run_model,
            ))
            scores[tid] = float(score.get("final_score", 0))
            _print_progress(tid, score, pred, failure)

    # 保存 trace
    trace_file = work_dir / f"trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
    trace_file.write_text(json.dumps({
        "suite": suite.suite_name, "agent": agent, "mode": "mock" if mock else "baseline",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _build_summary(results, scores)
    evaluation = {
        "suite_name": suite.suite_name,
        "agent": agent,
        "mode": "mock" if mock else "baseline",
        "results": results,
        "summary": summary,
    }
    (work_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTrace: {trace_file}\nEval: {work_dir / 'evaluation.json'}")
    return results


def _run_mock_task(loader, tid, task, knowledge_text: str):
    """mock 模式的 Worker 链：structured_query → python_analysis → answer（失败逐级回退）"""
    from workers.structured_query import StructuredQueryFlow
    from workers.python_analysis import PythonAnalysisFlow
    from workers.answer_worker import AnswerFlow

    trace = {"task_id": tid, "steps": [], "failure_reason": None, "succeeded": True}
    try:
        sq = StructuredQueryFlow()
        sq.set_para({"context_dir": str(task.context_dir), "user_request": task.question})
        sq_result = sq.kickoff()
        trace["steps"].append({"step": "structured_query", "ok": True})

        # analysis：LLM 生成 pandas 代码执行；失败则回退到 query 结果
        analysis_result = sq_result
        try:
            pa = PythonAnalysisFlow()
            pa.set_para({
                "context_dir": str(task.context_dir),
                "user_request": task.question,
                "context_preview": json.dumps(sq_result, ensure_ascii=False),
            })
            pa_result = pa.kickoff()
            trace["steps"].append({"step": "python_analysis", "ok": bool(pa_result.get("success"))})
            if pa_result.get("success") and pa_result.get("output"):
                analysis_result = {"structured_query": sq_result, "python_analysis": pa_result}
        except Exception:
            trace["steps"].append({"step": "python_analysis", "ok": False})

        aw = AnswerFlow()
        aw.set_para({
            "task_question": task.question,
            "analysis_result": json.dumps(analysis_result, ensure_ascii=False),
            "knowledge": knowledge_text,
        })
        answer = aw.kickoff()
        # 解析失败 / 空答案 → 视为未产出预测，避免空表头 CSV 被当成 success
        if not answer.get("columns") and not answer.get("rows"):
            trace["succeeded"] = False
            trace["failure_reason"] = answer.get("error") or "invalid_answer: 空答案"
            return trace, None
        pred = [answer.get("columns", [])] + answer.get("rows", [])
        trace["answer"] = answer
        return trace, pred
    except Exception as e:
        trace["succeeded"] = False
        trace["failure_reason"] = str(e)
        return trace, None


_ANSWER_TYPE_RE = re.compile(
    r"(numeric_exact|numeric_approximate|string_exact|string_approximate|list_exact|list_approximate)"
)


def _answer_type(knowledge: str) -> str:
    """从 task.knowledge 提取 KramaBench 的 answer_type（如 numeric_exact）。"""
    m = _ANSWER_TYPE_RE.search(knowledge or "")
    return m.group(1) if m else ""


def _score_task(task, pred, semantic: bool = False):
    """打分：KDD 表格走 ScoreFlow 行相等；单 cell 自由文本 benchmark 走 LLM 语义评分。
    无预测或 gold 缺失时给 0。score dict 只需含 final_score（0.99 阈值判 correct）。"""
    if semantic:
        from workers.judge_worker import JudgeFlow
        jf = JudgeFlow()
        jf.set_para({
            "question": task.question,
            "gold": task.gold_answer,
            "prediction": pred or [],
            "answer_type": _answer_type(task.knowledge),
        })
        try:
            return jf.kickoff()
        except Exception as e:
            return {"final_score": 0.0, "reason": str(e), "error": str(e)}
    from workers.score_worker import ScoreFlow
    sf = ScoreFlow()
    sf.set_para({"prediction": pred or [], "gold": task.gold_answer})
    try:
        return sf.kickoff()
    except Exception as e:
        return {"recall": 0.0, "redundancy_penalty": 0.0, "final_score": 0.0, "error": str(e)}


def _build_record(*, benchmark, agent, task_id, pred, trace, score,
                  task_dir, config_id, code_version, failure_hint, model="deepseek-chat") -> dict:
    """构造协议 §4 定义的 13 字段记录（不含嵌套 score）。"""
    return {
        "benchmark": benchmark,
        "agent": agent,
        "task_id": task_id,
        "status": "success" if pred else "failed",  # 运行成功 = 产出了预测（协议 §5）
        "prediction_path": str(task_dir / "prediction.csv") if pred else "",
        "trace_path": str(task_dir / "trace.json"),
        "steps": len(trace.get("steps", [])) if trace else 0,
        "latency_seconds": (trace or {}).get("e2e_elapsed_seconds", 0) or 0,
        "failure_type": _resolve_failure_type(pred, trace, score, failure_hint),
        "failure_reason": (trace or {}).get("failure_reason", "") if trace else "",
        "model": model,
        "config_id": config_id,
        "code_version": code_version,
    }


def _resolve_failure_type(pred, trace, score, failure_hint) -> str:
    """把预测/评分/trace 归类为协议 §5 的失败类型。"""
    if pred is None:
        # 未产出预测：优先用 trace 分类，其次按异常文本启发式归类
        if failure_hint:
            return failure_hint
        reason = ((trace or {}).get("failure_reason") or "").lower()
        if "api" in reason or "rate" in reason or "model" in reason:
            return "model_api"
        if "timeout" in reason:
            return "timeout"
        return "tool_error"
    if score.get("final_score", 0) < 0.99:
        return "wrong_answer"
    return ""


def _build_summary(results: list[dict], scores: dict[str, float]) -> dict:
    """汇总指标：区分运行成功率 / 答案正确率 / 基础设施失败率（协议 §12 验收标准）。"""
    total = len(results)
    run_success = sum(1 for r in results if r["status"] == "success")
    correct = sum(1 for tid in scores if scores.get(tid, 0) >= 0.99)
    wrong = sum(1 for r in results if r["failure_type"] == "wrong_answer")
    infra_failed = total - run_success
    avg = (sum(scores.values()) / total) if total else 0.0
    return {
        "total": total,
        "run_success": run_success,
        "run_success_rate": round(run_success / total, 4) if total else 0.0,
        "correct": correct,
        "correct_rate": round(correct / total, 4) if total else 0.0,
        "wrong_answer": wrong,
        "infra_failed": infra_failed,
        "infra_failed_rate": round(infra_failed / total, 4) if total else 0.0,
        "avg_final_score": round(avg, 4),
        "scores": scores,
    }


def _print_progress(tid, score, pred, failure=""):
    final = score.get("final_score", 0)
    if pred is None:
        emoji = "❌"
    elif final >= 0.99:
        emoji = "✅"
    else:
        emoji = "⚠️"
    extra = f" [{failure}]" if failure else ""
    print(f"  {emoji} {tid}: recall={score.get('recall',0):.3f} final={final:.3f} status={'success' if pred else 'failed'}{extra}")


def main():
    parser = argparse.ArgumentParser(description="统一 Data Agent 评测 Pipeline")
    parser.add_argument("--suite", default="pipeline_smoke_phase1_easy_3.json",
                        help="suite JSON 文件路径")
    parser.add_argument("--agent", default="react",
                        choices=["react", "dagent-lite", "agenticdata-lite", "mini-aop"])
    parser.add_argument("--benchmark", default="kdd",
                        choices=["kdd", "fdabench", "krama", "lakeqa"],
                        help="评测的 benchmark（决定用哪个 loader；默认 kdd）")
    parser.add_argument("--baseline-project", type=str,
                        help="Baseline 项目根目录（真实模式必需；用于定位 dabench 与 config）")
    parser.add_argument("--venv", type=str, default=None,
                        help="Baseline venv 目录名（默认自动探测 .venv-dagent/.venv）")
    parser.add_argument("--config", type=str, default=None,
                        help="Baseline config YAML 路径（默认自动探测 configs/ 下 *.local.yaml）")
    parser.add_argument("--task-timeout", type=int, default=600,
                        help="单任务超时秒数（写入 Baseline config，默认 600）")
    parser.add_argument("--mock", action="store_true",
                        help="mock 模式：用 Pipeline 自带 Worker 运行，不调用 Baseline")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = BASE / suite_path

    run_pipeline(
        suite_path=suite_path,
        agent=args.agent,
        mock=args.mock,
        benchmark=args.benchmark,
        baseline_project=Path(args.baseline_project) if args.baseline_project else None,
        venv_dir=args.venv,
        config_path=args.config,
        task_timeout=args.task_timeout,
    )


if __name__ == "__main__":
    main()
