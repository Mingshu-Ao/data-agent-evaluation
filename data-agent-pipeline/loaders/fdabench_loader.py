"""FDAbench benchmark 加载器 — 从 HuggingFace datasets 加载"""
import json
from pathlib import Path
from loaders.base import EvalTask


class FDAbenchLoader:
    """加载 FDAbench-Full 数据集（668 任务，task_id 为 FDA0001..FDA0668）。

    - `cache_dir`：HF 数据集缓存目录（只读；默认 ./data/fdabench，是符号链接）。
    - `context_root`：物化 context 的目录（Pipeline 自有，默认 ./data/fdabench_contexts）。
      不要把 context 写进共享的 HF 缓存/他人目录。
    """

    def __init__(self, cache_dir: str = "./data/fdabench", context_root: str = "./data/fdabench_contexts"):
        self.cache_dir = Path(cache_dir)
        self.context_root = Path(context_root)
        self._dataset = None

    def _load(self):
        if self._dataset is None:
            from datasets import load_dataset
            self._dataset = load_dataset(
                "FDAbench2026/FDAbench-Full",
                cache_dir=str(self.cache_dir), split="train",
            )
        return self._dataset

    def list_tasks(self) -> list[str]:
        ds = self._load()
        return [ds[i]["task_id"] for i in range(len(ds))]

    def task_count(self) -> dict:
        ds = self._load()
        levels = {}
        for i in range(len(ds)):
            lv = ds[i].get("level", "unknown")
            levels[lv] = levels.get(lv, 0) + 1
        return levels

    def load_task(self, task_id: str) -> EvalTask:
        ds = self._load()
        # Find by task_id
        row = None
        for i in range(len(ds)):
            if ds[i]["task_id"] == task_id:
                row = ds[i]
                break
        if row is None:
            raise ValueError(f"Task not found: {task_id}")

        context_dir = self.context_root / task_id / "context"
        context_dir.mkdir(parents=True, exist_ok=True)

        # 写 context 文件（preview 截断只用于元数据预览，不用于 gold）
        (context_dir / "task_meta.json").write_text(
            json.dumps({k: str(v)[:2000] for k, v in row.items() if v is not None},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if row.get("frozen_web_search"):
            (context_dir / "web_search.json").write_text(
                json.dumps(row["frozen_web_search"], ensure_ascii=False, indent=2), encoding="utf-8")
        if row.get("frozen_vector_search"):
            (context_dir / "vector_search.json").write_text(
                json.dumps(row["frozen_vector_search"], ensure_ascii=False, indent=2), encoding="utf-8")
        if row.get("rubric"):
            (context_dir / "rubric.json").write_text(
                json.dumps(row["rubric"], ensure_ascii=False, indent=2), encoding="utf-8")

        # gold：报告类用 ground_truth_report，SQL 类用 sql_result（不再截断，保证评分数据完整）
        gold = []
        if row.get("ground_truth_report"):
            gold = [["report"], [[row["ground_truth_report"]]]]
        elif row.get("sql_result"):
            gold = [["sql_result"], [[str(row["sql_result"])]]]

        return EvalTask(
            task_id=task_id,
            benchmark="fdabench",
            difficulty=row.get("level", "unknown"),
            question=str(row.get("query", "")),
            context_dir=context_dir,
            gold_answer=gold,
        )
