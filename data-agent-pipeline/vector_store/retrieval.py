"""非结构化数据 → 语义检索 编排层。

把 vector_store 的 document/image/video 处理 + VectorIndex 组合成一个函数，
供 workers（StructuredQueryFlow）在遇到非结构化文件时调用。
"""
from __future__ import annotations

from pathlib import Path

from vector_store.vector_index import VectorIndex

_DOC_EXTS = {".pdf", ".docx", ".doc", ".md", ".txt"}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _collect_nonstructured_files(context_dir: Path) -> list[Path]:
    """找出 context 里的非结构化文件（文档/图片/视频），忽略结构化数据文件"""
    out: list[Path] = []
    for path in context_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in _DOC_EXTS or ext in _IMG_EXTS or ext in _VID_EXTS:
            out.append(path)
    return out


def ingest_and_retrieve(
    context_dir: Path,
    query: str,
    top_k: int = 5,
    with_video: bool = True,
) -> list[dict]:
    """处理 context 中所有非结构化文件 → 建索引 → 按 query 返回 top_k 命中。

    返回 [{text, score, metadata: {source, type}}, ...]；无命中或出错返回 []。
    """
    files = _collect_nonstructured_files(context_dir)
    if not files:
        return []

    from vector_store.document_loader import DocumentLoader
    from vector_store.image_processor import ImageProcessor
    from vector_store.video_processor import VideoProcessor

    index = VectorIndex()
    doc_loader = DocumentLoader()
    img_processor = ImageProcessor()
    video_processor = VideoProcessor()

    for path in files:
        ext = path.suffix.lower()
        try:
            if ext in _DOC_EXTS:
                text = doc_loader.load(path)
                for chunk in doc_loader.chunk_text(text):
                    index.add_text(chunk, {"source": str(path), "type": "doc"})
            elif ext in _IMG_EXTS:
                index.add_text(img_processor.process(path), {"source": str(path), "type": "image"})
            elif with_video and ext in _VID_EXTS:
                index.add_text(video_processor.process(path), {"source": str(path), "type": "video"})
        except Exception:
            continue

    if not index:
        return []
    return index.search(query, top_k=top_k)
