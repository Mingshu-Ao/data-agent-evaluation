from __future__ import annotations

from pathlib import Path


def extract_pdf_text(path: Path, *, max_chars: int | None = None) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF support requires pypdf. Reinstall the project with `pip install -e .`."
        ) from exc

    reader = PdfReader(path)
    parts: list[str] = []
    current_length = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        current_length += len(text)
        if max_chars is not None and current_length >= max_chars:
            break
    joined = "\n".join(parts)
    return joined[:max_chars] if max_chars is not None else joined
