"""LakeQA benchmark 加载器

数据来源（ICML 2026，https://github.com/lakeagent/datalake-qa）：
  - 任务 JSON 在仓库的 lakeqa_mini/ 与 lakeqa_full/ 下：<k-*-d-*>/task_*.json
  - 数据文件在公共 S3 桶 lakeqa-yc4103-datalake，任务 JSON 的 datasets_used 字段列出所需文件
    （下载到本地 data/ 目录，保持与 datasets_used 相同的相对路径）

本地布局（data/lakeqa/）：
  lakeqa_mini/k-1-d-1/task_1.json   # git clone 的仓库
  lakeqa_full/...
  data/<datasets_used 相对路径>      # S3 下载的数据文件（scripts/download_lakeqa.py）

注意：LakeQA 答案是自由文本（如 "2021"），需要语义/LLM-judge 评分，
行相等评分（ScoreFlow）只能做记录，不能当最终评判。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from loaders.base import EvalTask


class LakeQALoader:
    """加载 LakeQA 任务（mini + full 两个 split 均支持）。"""

    SPLITS = ("lakeqa_mini", "lakeqa-full")

    def __init__(self, data_dir: str = "./data/lakeqa"):
        self.data_dir = Path(data_dir)
        self._tasks: Optional[dict[str, dict]] = None  # task_id -> task dict

    def _scan(self) -> None:
        if self._tasks is not None:
            return
        self._tasks = {}
        # 兼容两种布局：data_dir/<split>/ 或 data_dir/repo/<split>/（download_lakeqa.py clone 到 repo/）
        bases = [self.data_dir]
        repo_dir = self.data_dir / "repo"
        if repo_dir.is_dir():
            bases.append(repo_dir)
        for base in bases:
            for split in self.SPLITS:
                root = base / split
                if not root.is_dir():
                    continue
                for p in sorted(root.rglob("task_*.json")):
                    try:
                        with p.open(encoding="utf-8") as f:
                            t = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue
                    # question_id 在 mini/full 两个 split 间可能重叠（mini 是 full 子集），
                    # 用 "split:question_id" 做唯一 task_id
                    task_id = f"{split}:{str(t.get('question_id') or p.stem)}"
                    t["_path"] = str(p)
                    t["_split"] = split
                    self._tasks[task_id] = t

    def list_tasks(self) -> list[str]:
        self._scan()
        return sorted(self._tasks.keys())

    def task_count(self) -> dict:
        self._scan()
        counts = {}
        for _, t in self._tasks.items():
            key = f"{t.get('_split', '?')}:{len(t.get('datasets_used') or [])}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def load_task(self, task_id: str) -> EvalTask:
        self._scan()
        if task_id not in self._tasks:
            raise ValueError(f"Task not found: {task_id}")
        t = self._tasks[task_id]

        used = t.get("datasets_used") or []
        hops = t.get("reasoning_hops") or []
        knowledge = (
            f"split: {t.get('_split')}\n"
            f"所需数据文件（datasets_used）:\n"
            + "\n".join(f"  - {d}" for d in used[:30])
            + (f"\n推理跳数: {len(hops)}" if hops else "")
        ).strip()

        return EvalTask(
            task_id=task_id,
            benchmark="lakeqa",
            difficulty="unknown",
            question=str(t.get("question", "")),
            context_dir=self.data_dir / "data",  # S3 文件下载到此处（共享数据湖）
            knowledge=knowledge,
            gold_answer=[["answer"], [[str(t.get("answer", ""))]]],
        )
