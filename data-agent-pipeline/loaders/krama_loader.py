"""KramaBench benchmark 加载器（HF 镜像 eugenie-y/kramabench）

数据来源：https://huggingface.co/datasets/eugenie-y/kramabench
目录结构（snapshot_download 到 data/kramabench/）：
  workload/<domain>.json   # 每域任务清单（6 域：archeology/astronomy/biomedical/environment/legal/wildfire）
  data/<domain>/...        # 每域原始数据湖（共 ~1.7GB，1758 文件）
  solutions/<domain>/...   # 参考管线代码（104）

任务 JSON 字段：id / query / answer / answer_type / runtime / data_sources / subtasks
注意：datasets.load_dataset 对该仓库的类型推断有 bug（Float truncated to int64），
因此本 loader 直接读 workload/*.json，不做 datasets 加载。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from loaders.base import EvalTask

DOMAINS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]
HF_REPO = "eugenie-y/kramabench"


class KramaLoader:
    """加载 KramaBench（104 任务，6 域；data-lake 端到端数据管线类任务）。"""

    def __init__(self, data_dir: str = "./data/kramabench", auto_download: bool = True):
        self.data_dir = Path(data_dir)
        self.auto_download = auto_download
        self._tasks: Optional[dict[str, tuple[str, dict]]] = None  # task_id -> (domain, task)

    # ---------- 数据获取 ----------

    def _ensure_downloaded(self) -> None:
        """首次使用时从 HF snapshot_download（含 workload + data + solutions，~1.7GB）。"""
        workload_dir = self.data_dir / "workload"
        if workload_dir.is_dir() and any(workload_dir.glob("*.json")):
            return
        if not self.auto_download:
            raise FileNotFoundError(
                f"KramaBench 数据未下载（缺 {workload_dir}）。"
                f"请先运行 scripts/download_kramabench.py，或设置 auto_download=True。"
            )
        from huggingface_hub import snapshot_download
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[krama] 从 HF 下载 KramaBench 到 {self.data_dir}（含 data/ 约 1.7GB，首次较慢）...")
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(self.data_dir))

    def _load_all(self) -> None:
        if self._tasks is not None:
            return
        self._ensure_downloaded()
        self._tasks = {}
        for domain in DOMAINS:
            path = self.data_dir / "workload" / f"{domain}.json"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                tasks = json.load(f)
            for t in tasks:
                task_id = str(t.get("id") or "")
                if task_id:
                    self._tasks[task_id] = (domain, t)

    # ---------- 接口 ----------

    def list_tasks(self) -> list[str]:
        self._load_all()
        return sorted(self._tasks.keys())

    def task_count(self) -> dict:
        self._load_all()
        counts = {}
        for task_id, (_, t) in self._tasks.items():
            counts[self._difficulty(task_id)] = counts.get(self._difficulty(task_id), 0) + 1
        return counts

    @staticmethod
    def _difficulty(task_id: str) -> str:
        low = task_id.lower()
        if "-hard-" in low:
            return "hard"
        if "-medium-" in low:
            return "medium"
        return "easy"

    @staticmethod
    def _subtask_text(task: dict) -> str:
        subs = task.get("subtasks") or []
        if not subs:
            return ""
        lines = ["参考子任务："]
        for s in subs:
            q = str(s.get("query", "")).strip()
            if q:
                lines.append(f"- {q[:300]}")
        return "\n".join(lines)

    def load_task(self, task_id: str) -> EvalTask:
        self._load_all()
        if task_id not in self._tasks:
            raise ValueError(f"Task not found: {task_id}")
        domain, t = self._tasks[task_id]

        # 该任务引用的数据源（相对 domain data 目录），用于提示 Agent
        data_sources = t.get("data_sources") or []
        data_hint = "；".join(str(d) for d in data_sources[:20]) if data_sources else "(全部文件)"

        knowledge = (
            f"数据域: {domain}\n"
            f"所需数据源: {data_hint}\n"
            f"答案类型: {t.get('answer_type', 'unknown')}\n"
            f"{self._subtask_text(t)}"
        ).strip()

        return EvalTask(
            task_id=task_id,
            benchmark="krama",
            difficulty=self._difficulty(task_id),
            question=str(t.get("query", "")),
            context_dir=self.data_dir / "data" / domain,
            knowledge=knowledge,
            gold_answer=[["answer"], [[str(t.get("answer"))]]],
        )
