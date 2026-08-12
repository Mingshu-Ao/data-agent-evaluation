"""图片处理器：OCR + VLM caption → 文本描述"""
from pathlib import Path


class ImageProcessor:
    """处理图片：OCR 提取文字 + VLM 生成描述"""

    @staticmethod
    def ocr(path: Path) -> str:
        """OCR 提取图片文字"""
        try:
            from paddleocr import PaddleOCR
            ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            result = ocr_engine.ocr(str(path))
            lines = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0] if len(line) > 1 else ""
                    if text:
                        lines.append(text)
            return "\n".join(lines)
        except ImportError:
            return f"[OCR not available for: {path.name}]"

    @staticmethod
    def caption(path: Path) -> str:
        """VLM 生成图片描述（通过 MCP 视觉工具）"""
        # TODO: 集成 MCP 视觉工具
        return f"[VLM caption pending for: {path.name}]"

    def process(self, path: Path) -> str:
        """OCR + VLM 双路处理，合并文本"""
        ocr_text = self.ocr(path)
        caption_text = self.caption(path)
        return f"IMAGE: {path.name}\nOCR:\n{ocr_text}\nVLM:\n{caption_text}"
