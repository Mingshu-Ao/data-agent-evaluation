"""文档处理器：PDF/Word/Markdown → 文本 → 分块"""
from pathlib import Path


class DocumentLoader:
    """提取文档文本并分块，准备写入向量库。"""

    @staticmethod
    def load_pdf(path: Path) -> str:
        """PDF → 文本（用 PyMuPDF）"""
        try:
            import fitz
            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            # 降级：读原始文本
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def load_docx(path: Path) -> str:
        """Word → 文本"""
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def load_markdown(path: Path) -> str:
        """Markdown → 文本"""
        return path.read_text(encoding="utf-8", errors="replace")

    def load(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(path)
        elif ext in (".docx", ".doc"):
            return self.load_docx(path)
        elif ext in (".md", ".txt"):
            return self.load_markdown(path)
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """将长文本分成重叠块"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
