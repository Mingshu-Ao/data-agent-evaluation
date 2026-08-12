"""结构化数据查询 Worker"""
from __future__ import annotations
import csv
import json
import sqlite3
from pathlib import Path
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel
from TAS.Crewai.utils import AgentFlow
from eval_config import WORKER_MODEL, WORKER_BASE_URL, WORKER_API_KEY


class StructuredQueryState(BaseModel):
    context_dir: str = ""
    user_request: str = ""
    result: dict = {}


class StructuredQueryFlow(AgentFlow[StructuredQueryState]):
    name: str = "StructuredQuery"
    description: str = (
        "查询结构化数据目录中的文件：列出所有 CSV/JSON/SQLite 文件，"
        "根据请求读取文件内容。返回文件列表和数据预览。"
    )
    llm = LLM(model=WORKER_MODEL, base_url=WORKER_BASE_URL, api_key=WORKER_API_KEY)

    def to_param(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_dir": {"type": "string", "description": "数据目录的绝对路径"},
                        "user_request": {"type": "string", "description": "用户的查询请求"},
                    },
                    "required": ["context_dir", "user_request"],
                },
            },
        }

    def set_para(self, para: dict):
        self.state.context_dir = para.get("context_dir", "")
        self.state.user_request = para.get("user_request", "")

    def _list_files(self, ctx: Path) -> list:
        files = []
        for f in sorted(ctx.rglob("*")):
            if f.is_file():
                files.append(str(f.relative_to(ctx)))
        return files

    def _read_csv(self, path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            reader = list(csv.reader(f))
        return {"columns": reader[0] if reader else [], "rows": reader[1:21], "total_rows": len(reader) - 1}

    def _read_json(self, path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        preview = json.dumps(data, ensure_ascii=False)[:2000]
        return {"preview": preview, "truncated": len(str(data)) > 2000}

    def _read_sqlite(self, path: Path) -> dict:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        return {"tables": [{"name": t[0], "sql": t[1]} for t in tables]}

    def _read_doc(self, path: Path) -> dict:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {"preview": text[:2000], "truncated": len(text) > 2000}

    @start()
    def explore_and_query(self):
        ctx = Path(self.state.context_dir)
        if not ctx.exists():
            self.state.result = {"error": f"目录不存在: {ctx}"}
            return self.state.result

        files = self._list_files(ctx)
        result = {"root": str(ctx), "files": files, "data": {}}
        for f in files:
            path = ctx / f
            ext = path.suffix.lower()
            try:
                if ext == ".csv":
                    result["data"][f] = self._read_csv(path)
                elif ext == ".json":
                    result["data"][f] = self._read_json(path)
                elif ext in (".db", ".sqlite", ".sqlite3"):
                    result["data"][f] = self._read_sqlite(path)
                elif ext in (".md", ".txt"):
                    result["data"][f] = self._read_doc(path)
            except Exception as e:
                result["data"][f] = {"error": str(e)}

        # 非结构化文件（PDF/图片/视频）→ 语义检索（可选；失败不影响主流程）
        try:
            from vector_store.retrieval import ingest_and_retrieve
            hits = ingest_and_retrieve(ctx, self.state.user_request, top_k=3)
            if hits:
                result["semantic_retrieval"] = hits
        except Exception:
            pass

        self.state.result = result
        return result
