from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


def find_first_video(context_dir: Path) -> Path | None:
    for path in sorted(context_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def find_attached_keyframes(context_dir: Path) -> list[Path]:
    manifest_path = context_dir / "video_evidence" / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not manifest.get("attach_keyframes"):
        return []

    keyframes: list[Path] = []
    for frame in manifest.get("frames", []):
        if not isinstance(frame, dict):
            continue
        relative_path = frame.get("image_path")
        if not isinstance(relative_path, str):
            continue
        image_path = (context_dir / relative_path).resolve()
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
            and context_dir.resolve() in image_path.parents
        ):
            keyframes.append(image_path)
    return keyframes


def _data_url(path: Path, fallback_mime_type: str) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or fallback_mime_type
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_user_content_with_video(
    task: PublicTask,
    text: str,
) -> str | list[dict[str, Any]]:
    video_path = find_first_video(task.context_dir)
    if video_path is not None:
        relative_path = video_path.relative_to(task.context_dir).as_posix()
        return [
            {
                "type": "text",
                "text": f"{text}\n\nAttached context video: {relative_path}",
            },
            {
                "type": "video_url",
                "video_url": {"url": _data_url(video_path, "video/mp4")},
            },
        ]

    keyframes = find_attached_keyframes(task.context_dir)
    if not keyframes:
        return text
    resolved_context_dir = task.context_dir.resolve()
    relative_paths = [path.relative_to(resolved_context_dir).as_posix() for path in keyframes]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{text}\n\nAttached {len(keyframes)} visual-change keyframes: "
                f"{', '.join(relative_paths)}"
            ),
        }
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _data_url(path, "image/jpeg")},
        }
        for path in keyframes
    )
    return content
