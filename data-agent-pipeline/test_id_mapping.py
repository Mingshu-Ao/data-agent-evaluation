"""TaskIdMapper 单测：恒等 / 映射 / 回逆 / 碰撞避让（无 API、无网络）。

用法:
    .venv\\Scripts\\python.exe test_id_mapping.py
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from baseline_adapter import TaskIdMapper


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {name}")
    print(f"  [ok] {name}")


def main() -> None:
    # 1) 混合 suite：KDD 恒等 + 三个 benchmark 顺序映射
    mapper = TaskIdMapper(["task_11", "FDA0001", "legal-hard-1", "lakeqa-full:EQA000001"])
    _check("task_11 恒等", mapper.to_baseline("task_11") == "task_11")
    _check("FDA0001 → task_100000", mapper.to_baseline("FDA0001") == "task_100000")
    _check("legal-hard-1 → task_100001", mapper.to_baseline("legal-hard-1") == "task_100001")
    _check("lakeqa-full:EQA000001 → task_100002", mapper.to_baseline("lakeqa-full:EQA000001") == "task_100002")
    _check("回逆 FDA0001", mapper.from_baseline("task_100000") == "FDA0001")
    _check("回逆 legal-hard-1", mapper.from_baseline("task_100001") == "legal-hard-1")
    _check("回逆 task_11", mapper.from_baseline("task_11") == "task_11")
    _check("未知 ID 原样返回", mapper.to_baseline("nope") == "nope")

    # 2) 碰撞避让：已有 task_100000 时，FDA0001 应跳开
    mapper2 = TaskIdMapper(["task_100000", "FDA0001"])
    _check("task_100000 恒等", mapper2.to_baseline("task_100000") == "task_100000")
    _check("FDA0001 避开 100000 → task_100001", mapper2.to_baseline("FDA0001") == "task_100001")
    _check("碰撞回逆 FDA0001", mapper2.from_baseline("task_100001") == "FDA0001")

    # 3) 双向一致 + 恒等保留原 ID、非 task_<int> 映射到 ≥100000
    ids = ["FDA0001", "legal-hard-1", "lakeqa-full:EQA000001", "lakeqa_mini:EQA000002",
           "archeology-easy-3", "wildfire-hard-7", "task_24"]
    m3 = TaskIdMapper(ids)
    for tid in ids:
        mapped = m3.to_baseline(tid)
        _check(f"roundtrip {tid}", m3.from_baseline(mapped) == tid)
        num = mapped[len("task_"):]
        assert mapped.startswith("task_") and num.isdigit(), mapped
        if tid.startswith("task_"):
            _check(f"恒等保留 {tid}", mapped == tid)
        else:
            _check(f"映射 ≥100000 {tid}", int(num) >= 100000)

    print("\nTaskIdMapper: ALL PASS")


if __name__ == "__main__":
    main()
