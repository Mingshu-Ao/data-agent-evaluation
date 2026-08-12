"""JudgeFlow 真实 API 冒烟：exact 数字 / 错误答案 / approximate 语义等价。

需要 Pipeline venv（crewai + TAS 可导入）+ 真实 API key（eval_config 环境变量或 fallback）。

用法:
    .venv\\Scripts\\python.exe test_judge_smoke.py
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def _ensure_tas_importable() -> None:
    """与 eval_pipeline._ensure_tas_importable 等价的轻量探测（TAS 父目录加入 sys.path）。"""
    import importlib.util  # noqa: PLC0415
    if importlib.util.find_spec("TAS") is not None:
        return
    import os  # noqa: PLC0415
    candidates = [Path(os.environ.get("TAS_PATH", "")), Path(os.environ.get("TAS_PATH", "")) / "TAS"]
    candidates += [BASE.parent / "data_agent" / "miaohongfan", BASE.parent / "miaohongfan"]
    for parent in candidates:
        if (parent / "TAS").is_dir() and str(parent) not in sys.path:
            sys.path.append(str(parent))
            return
    print("[warn] 未找到 TAS 包（JudgeFlow 依赖 TAS.Crewai.utils），judge 冒烟可能失败")


def run_case(question: str, gold_table: list, pred_table: list, answer_type: str, label: str) -> dict:
    from workers.judge_worker import JudgeFlow
    jf = JudgeFlow()
    jf.set_para({"question": question, "gold": gold_table,
                 "prediction": pred_table, "answer_type": answer_type})
    res = jf.kickoff()
    print(f"  {label}: final={res.get('final_score', 0):.2f} "
          f"reason={str(res.get('reason', ''))[:100]}")
    return res


def main() -> None:
    _ensure_tas_importable()

    # 1) numeric_exact：数字完全一致 → ≥0.99
    r1 = run_case(
        "该数据集里 2019 年记录的总数是多少？",
        [["answer"], [["13427"]]], [["answer"], [["13427"]]],
        "numeric_exact", "exact-number")

    # 2) string_exact：错误年份 → <0.99
    r2 = run_case(
        "某公司成立于哪一年？",
        [["answer"], [["2020"]]], [["answer"], [["2019"]]],
        "string_exact", "wrong-year")

    # 3) string_approximate：语义等价 → ≥0.99
    r3 = run_case(
        "该地区属于哪个大都市统计区？",
        [["answer"], [["Atlanta-Sandy Springs-Roswell GA Metropolitan Statistical Area"]]],
        [["answer"], [["Atlanta metropolitan area"]]],
        "string_approximate", "approx-metro")

    # 4) numeric_approximate：数值接近 → ≥0.99
    r4 = run_case(
        "该数据集的平均房价是多少？",
        [["answer"], [["13427.5676"]]], [["answer"], [["13427"]]],
        "numeric_approximate", "approx-number")

    ok = (r1.get("final_score", 0) >= 0.99 and r2.get("final_score", 0) < 0.99
          and r3.get("final_score", 0) >= 0.99 and r4.get("final_score", 0) >= 0.99)
    print("JudgeFlow smoke:", "PASS" if ok else "FAIL (review scores above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
