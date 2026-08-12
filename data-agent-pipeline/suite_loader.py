"""任务清单加载器：读取 suite JSON 文件"""
import json
from pathlib import Path
from typing import Optional


class SuiteLoader:
    """加载 suite JSON，返回任务 ID 列表和元信息。

    suite JSON 结构：
      suite_name, description, task_ids: list[str],
      tasks: list[{task_id, difficulty, ...}], suite_summary
    """

    def __init__(self, suite_path: Path):
        self.path = Path(suite_path)
        with self.path.open(encoding="utf-8") as f:
            self.data = json.load(f)
        self._validate()

    def _validate(self):
        if not isinstance(self.data, dict):
            raise ValueError(f"suite JSON 顶层必须是对象: {self.path}")
        ids = self.data.get("task_ids", [])
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"suite 缺少非空 task_ids: {self.path}")
        for tid in ids:
            if not isinstance(tid, str):
                raise ValueError(f"task_ids 中存在非字符串元素: {self.path}")
        # tasks 元信息与 task_ids 的一致性检查
        seen = {t.get("task_id") for t in self.data.get("tasks", []) if isinstance(t, dict)}
        missing = [t for t in ids if t not in seen]
        extra = [t for t in seen if t not in ids] if seen else []
        if missing:
            print(f"[warn] suite 的 tasks 元信息缺少以下 task_id: {missing}")

    @property
    def suite_name(self) -> str:
        return self.data.get("suite_name", self.path.stem)

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    def task_ids(self) -> list[str]:
        """返回任务 ID 列表"""
        return list(self.data.get("task_ids", []))

    def task_info(self, task_id: str) -> Optional[dict]:
        """返回单个任务的元信息"""
        for t in self.data.get("tasks", []):
            if isinstance(t, dict) and t.get("task_id") == task_id:
                return t
        return None

    def summary(self) -> dict:
        return self.data.get("suite_summary", {})
