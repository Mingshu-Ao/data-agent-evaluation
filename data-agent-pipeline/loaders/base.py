"""统一 EvalTask 数据格式 — 所有 benchmark 都转成这个格式"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalTask:
    task_id: str
    benchmark: str          # "kdd" | "fdabench" | "krama" | "lakeqa"
    difficulty: str
    question: str
    context_dir: Path       # 结构化数据目录 (CSV/JSON/SQLite)
    knowledge: str = ""     # 非结构化文本 / Markdown
    gold_answer: list = field(default_factory=list)  # gold.csv 内容

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "difficulty": self.difficulty,
            "question": self.question,
            "context_dir": str(self.context_dir),
            "knowledge": self.knowledge,
            "gold_answer": self.gold_answer,
        }
