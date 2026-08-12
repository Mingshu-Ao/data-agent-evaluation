"""轻量自包含向量索引（无需 chromadb / langchain / 外部 embedding 模型）。

设计原因：chromadb 与 crewai 的 pydantic v2 存在版本冲突风险，且 DashScopeEmbeddings
需要额外 key。这里用「TF 哈希 embedding + 余弦相似度」：
  - token 用 hashlib 稳定哈希（跨进程确定，不受 PYTHONHASHSEED 影响）
  - 中文/英文/数字 token 都覆盖（\\w 在 Python 3 下匹配 CJK）
  - 纯 Python 实现，零额外依赖；向量库持久化为 JSON

检索质量：TF-hash 是词袋近似，弱于语义 embedding，但确定、离线、够用；
如需更强语义，可自行替换 embedding 函数。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_LATIN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]")


def _tokens(text: str) -> list[str]:
    """中英文混合分词：英文/数字按单词，中文按字符 unigram + bigram。"""
    lower = text.lower()
    toks = _LATIN_RE.findall(lower)
    cjk = _CJK_RE.findall(text)
    toks += cjk
    toks += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return toks


def _token_hash(token: str, dim: int) -> int:
    """稳定哈希：token -> [0, dim)"""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim


def embed_text(text: str, dim: int = 512) -> list[float]:
    """TF 哈希 embedding + L2 归一化"""
    vec = [0.0] * dim
    for tok in _tokens(text):
        vec[_token_hash(tok, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """管理文档的向量存储和检索。数据保存在内存 + 可持久化到 JSON。"""

    def __init__(self, persist_dir: str = "./vector_db", dim: int = 512):
        self.persist_dir = Path(persist_dir)
        self.dim = dim
        self._docs: list[dict] = []  # {id, text, metadata, vector}

    def __len__(self) -> int:
        return len(self._docs)

    # ---------- 写入 ----------

    def add_documents(self, texts: list[str], metadata: list[dict] | None = None):
        metadata = metadata or [{}] * len(texts)
        for i, text in enumerate(texts):
            doc_id = hashlib.md5(f"{len(self._docs)}|{text[:200]}".encode()).hexdigest()[:16]
            self._docs.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata[i] if i < len(metadata) else {},
                "vector": embed_text(text, self.dim),
            })

    def add_text(self, text: str, metadata: dict | None = None):
        self.add_documents([text], [metadata or {}])

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索，返回 [{text, score, metadata}, ...]"""
        if not self._docs:
            return []
        qv = embed_text(query, self.dim)
        scored = [(cosine(qv, d["vector"]), d) for d in self._docs]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            {"text": d["text"], "score": round(s, 4), "metadata": d["metadata"]}
            for s, d in scored[:top_k]
        ]

    # ---------- 持久化 ----------

    def save(self, name: str = "index.json"):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "docs": [
                {"id": d["id"], "text": d["text"], "metadata": d["metadata"], "vector": d["vector"]}
                for d in self._docs
            ],
        }
        (self.persist_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, name: str = "index.json"):
        path = self.persist_dir / name
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.dim = int(payload.get("dim", self.dim))
        self._docs = list(payload.get("docs", []))

    # ---------- 批量导入 ----------

    def ingest_directory(self, dir_path: Path):
        """批量导入目录中的非结构化文件（文档/图片/视频）"""
        from vector_store.document_loader import DocumentLoader
        from vector_store.image_processor import ImageProcessor
        from vector_store.video_processor import VideoProcessor

        doc_loader = DocumentLoader()
        img_processor = ImageProcessor()
        video_processor = VideoProcessor()

        for path in dir_path.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            texts: list[str] = []
            try:
                if ext in (".pdf", ".docx", ".doc", ".md", ".txt"):
                    text = doc_loader.load(path)
                    texts = doc_loader.chunk_text(text)
                elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    texts = [img_processor.process(path)]
                elif ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    texts = [video_processor.process(path)]
            except Exception:
                continue
            if texts:
                self.add_documents(texts, [{"source": str(path)}] * len(texts))
