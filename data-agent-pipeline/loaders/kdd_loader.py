"""KDD Cup benchmark 加载器"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from loaders.base import EvalTask
from typing import Optional


class KDDLoader:
    """加载 KDD Cup Phase 1 或 Phase 2 数据集"""

    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else None

    def list_tasks(self) -> list[str]:
        return sorted(
            d.name for d in self.input_dir.iterdir()
            if d.is_dir() and d.name.startswith("task_")
        )

    def task_count(self) -> dict:
        """统计各难度的任务数量"""
        counts = {}
        for tid in self.list_tasks():
            task = self.load_task(tid)
            counts[task.difficulty] = counts.get(task.difficulty, 0) + 1
        return counts

    def load_task(self, task_id: str) -> EvalTask:
        task_dir = self.input_dir / task_id

        with (task_dir / "task.json").open(encoding="utf-8") as f:
            meta = json.load(f)

        knowledge = ""
        km_path = task_dir / "context" / "knowledge.md"
        if km_path.exists():
            knowledge = km_path.read_text(encoding="utf-8")

        gold = []
        if self.output_dir:
            gold_path = self.output_dir / task_id / "gold.csv"
            if gold_path.exists():
                with gold_path.open(encoding="utf-8") as f:
                    gold = list(csv.reader(f))

        return EvalTask(
            task_id=task_id,
            benchmark="kdd",
            difficulty=meta.get("difficulty", "unknown"),
            question=meta.get("question", ""),
            context_dir=task_dir / "context",
            knowledge=knowledge,
            gold_answer=gold,
        )
