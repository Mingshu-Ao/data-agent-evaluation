from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from data_agent_baseline.agents.multimodal import VIDEO_EXTENSIONS
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.suites import load_suite_task_ids

VIDEO_EVIDENCE_MODES = {
    "ocr",
    "ocr_keyframes",
    "ocr_keyframes_asr",
}


def sample_video_timestamps(
    duration_seconds: float,
    *,
    interval_seconds: float = 10.0,
    max_frames: int = 12,
) -> list[float]:
    if duration_seconds <= 0:
        return [0.0]
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    start = min(5.0, max(0.5, duration_seconds * 0.05))
    end = max(start, duration_seconds - 1.0)
    natural_count = max(1, math.floor((end - start) / interval_seconds) + 1)
    count = min(max_frames, natural_count)
    if count == 1:
        return [round(start, 3)]
    if natural_count <= max_frames:
        return [round(min(start + index * interval_seconds, end), 3) for index in range(count)]
    spacing = (end - start) / (max_frames - 1)
    return [round(start + index * spacing, 3) for index in range(max_frames)]


def visual_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        raise ValueError("Visual signatures must have matching shapes.")
    difference = np.abs(current.astype(np.float32) - previous.astype(np.float32)) / 255.0
    mean_difference = float(difference.mean())
    changed_fraction = float((difference >= 0.10).mean())
    high_difference = float(np.quantile(difference, 0.95))
    return round(
        0.45 * mean_difference + 0.35 * changed_fraction + 0.20 * high_difference,
        6,
    )


def select_visual_change_timestamps(
    candidates: list[tuple[float, float]],
    *,
    duration_seconds: float,
    interval_seconds: float,
    max_frames: int,
    minimum_change_score: float,
    minimum_gap_seconds: float,
) -> list[float]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if minimum_change_score < 0:
        raise ValueError("minimum_change_score must be non-negative.")
    if minimum_gap_seconds < 0:
        raise ValueError("minimum_gap_seconds must be non-negative.")

    fallback = sample_video_timestamps(
        duration_seconds,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    if not candidates:
        return fallback

    selected = [round(candidates[0][0], 3)]
    ranked_candidates = sorted(candidates[1:], key=lambda item: item[1], reverse=True)

    def add_candidates(items: list[tuple[float, float]]) -> None:
        for timestamp, _ in items:
            if len(selected) >= max_frames:
                return
            if all(abs(timestamp - existing) >= minimum_gap_seconds for existing in selected):
                selected.append(round(timestamp, 3))

    if len(candidates) > 1:
        add_candidates([candidates[-1]])

    change_frame_budget = min(max_frames, max(3, math.ceil(max_frames * 0.65)))
    high_change_candidates = [
        item for item in ranked_candidates if item[1] >= minimum_change_score
    ]
    add_candidates(high_change_candidates[:change_frame_budget])
    if len(selected) < change_frame_budget:
        add_candidates(ranked_candidates)

    # Scene scores alone miss semantically important slides with subtle visual changes.
    # Fill the remaining budget with uniform anchors to retain full timeline coverage.
    add_candidates([(timestamp, 0.0) for timestamp in fallback])
    add_candidates(ranked_candidates)

    return sorted(selected[:max_frames])


def _video_dependencies() -> tuple[Any, Any]:
    try:
        import av
        from rapidocr_onnxruntime import RapidOCR
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Video preprocessing requires optional dependencies. Install with "
            "`python -m pip install -e .[video]`."
        ) from exc
    return av, RapidOCR


def _asr_dependency() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ASR preprocessing requires optional dependencies. Install with "
            "`python -m pip install -e .[video-asr]`."
        ) from exc
    return WhisperModel


def _visual_signature(image: Any) -> np.ndarray:
    grayscale = image.convert("L").resize((96, 54))
    return np.asarray(grayscale, dtype=np.uint8)


def detect_visual_change_timestamps(
    *,
    video_path: Path,
    duration_seconds: float,
    interval_seconds: float,
    max_frames: int,
    probe_interval_seconds: float,
    minimum_change_score: float,
    minimum_gap_seconds: float,
) -> tuple[list[float], list[dict[str, float]]]:
    if probe_interval_seconds <= 0:
        raise ValueError("probe_interval_seconds must be positive.")

    av, _ = _video_dependencies()
    candidates: list[tuple[float, float]] = []
    previous_signature: np.ndarray | None = None
    start_seconds = min(5.0, max(0.5, duration_seconds * 0.05))
    end_seconds = max(start_seconds, duration_seconds - 1.0)
    next_probe_seconds = start_seconds
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video has no video stream: {video_path}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 1e-6 < next_probe_seconds:
                continue
            if timestamp > end_seconds:
                break
            signature = _visual_signature(frame.to_image())
            score = (
                0.0
                if previous_signature is None
                else visual_change_score(previous_signature, signature)
            )
            candidates.append((timestamp, score))
            previous_signature = signature
            next_probe_seconds = timestamp + probe_interval_seconds

    timestamps = select_visual_change_timestamps(
        candidates,
        duration_seconds=duration_seconds,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        minimum_change_score=minimum_change_score,
        minimum_gap_seconds=minimum_gap_seconds,
    )
    scores = [
        {
            "timestamp_seconds": round(timestamp, 3),
            "change_score": round(score, 6),
        }
        for timestamp, score in candidates
    ]
    return timestamps, scores


def transcribe_video_audio(
    *,
    video_path: Path,
    model: Any,
    language: str | None,
) -> dict[str, Any]:
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    transcript_segments = [
        {
            "start_seconds": round(float(segment.start), 3),
            "end_seconds": round(float(segment.end), 3),
            "text": str(segment.text).strip(),
        }
        for segment in segments
        if str(segment.text).strip()
    ]
    return {
        "language": str(getattr(info, "language", language or "unknown")),
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0)),
            4,
        ),
        "segments": transcript_segments,
    }


def _decode_frame(container: Any, stream: Any, target_seconds: float) -> tuple[Any, float] | None:
    time_base = float(stream.time_base)
    container.seek(
        max(int(target_seconds / time_base), 0),
        stream=stream,
        backward=True,
    )
    last_frame = None
    last_timestamp = target_seconds
    for frame in container.decode(stream):
        last_frame = frame
        if frame.pts is None:
            continue
        last_timestamp = float(frame.pts * stream.time_base)
        if last_timestamp >= target_seconds:
            return frame, last_timestamp
    if last_frame is None:
        return None
    return last_frame, last_timestamp


def _ocr_frame(
    engine: Any,
    image: Any,
    *,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    result, _ = engine(image)
    image_array = np.asarray(image.convert("RGB"))
    lines: list[dict[str, Any]] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        confidence = float(item[2])
        if text and confidence >= minimum_confidence:
            box = np.asarray(item[0], dtype=int)
            x_min = max(int(box[:, 0].min()) - 10, 0)
            x_max = min(int(box[:, 0].max()) + 11, image_array.shape[1])
            y_min = max(int(box[:, 1].min()) - 10, 0)
            y_max = min(int(box[:, 1].max()) + 11, image_array.shape[0])
            region = image_array[y_min:y_max, x_min:x_max].astype(int)
            channel_spread = region.max(axis=2) - region.min(axis=2)
            blue_pixels = (
                (region[:, :, 2] > region[:, :, 0] + 25)
                & (region[:, :, 2] > region[:, :, 1] + 15)
                & (channel_spread > 45)
            )
            blue_background_fraction = round(float(blue_pixels.mean()), 4) if region.size else 0.0
            lines.append(
                {
                    "text": text,
                    "confidence": round(confidence, 4),
                    "visually_highlighted": blue_background_fraction >= 0.25,
                    "blue_background_fraction": blue_background_fraction,
                }
            )
    return lines


def render_video_evidence(
    *,
    source_video: str,
    duration_seconds: float,
    frames: list[dict[str, Any]],
    audio_stream_count: int,
    sampling_strategy: str = "uniform",
    transcript: dict[str, Any] | None = None,
) -> str:
    transcript_segments = (transcript or {}).get("segments", [])
    lines = [
        "# Video Evidence",
        "",
        (
            "This file was generated locally from sampled video frames. "
            "OCR text may contain recognition errors; verify thresholds, dates, field names, "
            "deduplication rules, and grouping dimensions against multiple nearby frames."
        ),
        "",
        f"- Source video: `{source_video}`",
        f"- Duration: {duration_seconds:.2f} seconds",
        f"- Sampled frames: {len(frames)}",
        f"- Frame sampling strategy: {sampling_strategy}",
        f"- Audio streams detected: {audio_stream_count}",
        (
            f"- Audio transcription: {len(transcript_segments)} timestamped segments"
            if transcript_segments
            else "- Audio transcription: not generated"
        ),
        "",
    ]
    for frame in frames:
        lines.extend(
            [
                f"## Frame {frame['frame_index']} at {frame['timestamp_seconds']:.2f}s",
                "",
                f"- Image: `{frame['image_path']}`",
            ]
        )
        ocr_lines = frame.get("ocr_lines", [])
        if ocr_lines:
            lines.append("- OCR text:")
            lines.extend(
                (
                    f"  - {item['text']} [VISUALLY HIGHLIGHTED / LIKELY SELECTED]"
                    if item.get("visually_highlighted")
                    else f"  - {item['text']}"
                )
                for item in ocr_lines
            )
        else:
            lines.append("- OCR text: none detected")
        lines.append("")
    if transcript_segments:
        lines.extend(
            [
                "## ASR Transcript",
                "",
                (
                    f"- Detected language: {transcript.get('language', 'unknown')} "
                    f"(probability {transcript.get('language_probability', 0.0):.2f})"
                ),
                "",
            ]
        )
        lines.extend(
            (
                f"- [{segment['start_seconds']:.2f}s - "
                f"{segment['end_seconds']:.2f}s] {segment['text']}"
            )
            for segment in transcript_segments
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preprocess_video(
    *,
    video_path: Path,
    output_context_dir: Path,
    ocr_engine: Any,
    interval_seconds: float,
    max_frames: int,
    minimum_confidence: float,
    evidence_mode: str = "ocr",
    scene_probe_interval_seconds: float = 1.0,
    scene_change_threshold: float = 0.12,
    scene_min_gap_seconds: float = 2.0,
    asr_model: Any | None = None,
    asr_language: str | None = None,
    attach_keyframes: bool = True,
) -> dict[str, Any]:
    if evidence_mode not in VIDEO_EVIDENCE_MODES:
        raise ValueError(
            f"Unsupported evidence_mode {evidence_mode!r}; "
            f"choose one of {sorted(VIDEO_EVIDENCE_MODES)}."
        )
    av, _ = _video_dependencies()
    evidence_dir = output_context_dir / "video_evidence"
    frames_dir = evidence_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, Any]] = []
    visual_change_scores: list[dict[str, float]] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video has no video stream: {video_path}")
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        else:
            raise ValueError(f"Unable to determine video duration: {video_path}")
        if evidence_mode == "ocr":
            sampling_strategy = "uniform"
            timestamps = sample_video_timestamps(
                duration_seconds,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
            )
        else:
            sampling_strategy = "visual_change"
            timestamps, visual_change_scores = detect_visual_change_timestamps(
                video_path=video_path,
                duration_seconds=duration_seconds,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
                probe_interval_seconds=scene_probe_interval_seconds,
                minimum_change_score=scene_change_threshold,
                minimum_gap_seconds=scene_min_gap_seconds,
            )
        for index, target_seconds in enumerate(timestamps, start=1):
            decoded = _decode_frame(container, stream, target_seconds)
            if decoded is None:
                continue
            frame, actual_timestamp = decoded
            image = frame.to_image()
            image_name = f"frame_{index:03d}_{actual_timestamp:07.2f}s.jpg"
            image_path = frames_dir / image_name
            image.save(image_path, quality=90)
            frame_records.append(
                {
                    "frame_index": index,
                    "target_seconds": target_seconds,
                    "timestamp_seconds": round(actual_timestamp, 3),
                    "image_path": f"video_evidence/frames/{image_name}",
                    "ocr_lines": _ocr_frame(
                        ocr_engine,
                        image,
                        minimum_confidence=minimum_confidence,
                    ),
                }
            )
        audio_stream_count = len(container.streams.audio)
        metadata = {
            "width": stream.width,
            "height": stream.height,
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
            "duration_seconds": round(duration_seconds, 3),
            "audio_stream_count": audio_stream_count,
        }

    transcript: dict[str, Any] | None = None
    if evidence_mode == "ocr_keyframes_asr" and audio_stream_count > 0:
        if asr_model is None:
            raise ValueError("asr_model is required for ocr_keyframes_asr mode.")
        transcript = transcribe_video_audio(
            video_path=video_path,
            model=asr_model,
            language=asr_language,
        )
        (evidence_dir / "asr_transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    source_video = video_path.name
    evidence_text = render_video_evidence(
        source_video=source_video,
        duration_seconds=duration_seconds,
        frames=frame_records,
        audio_stream_count=audio_stream_count,
        sampling_strategy=sampling_strategy,
        transcript=transcript,
    )
    evidence_path = output_context_dir / "video_evidence.md"
    evidence_path.write_text(evidence_text, encoding="utf-8")
    manifest = {
        "source_video": str(video_path),
        "evidence_mode": evidence_mode,
        "attach_keyframes": evidence_mode != "ocr" and attach_keyframes,
        "asr_enabled": evidence_mode == "ocr_keyframes_asr",
        "metadata": metadata,
        "sampling": {
            "strategy": sampling_strategy,
            "interval_seconds": interval_seconds,
            "max_frames": max_frames,
            "minimum_ocr_confidence": minimum_confidence,
            "scene_probe_interval_seconds": scene_probe_interval_seconds,
            "scene_change_threshold": scene_change_threshold,
            "scene_min_gap_seconds": scene_min_gap_seconds,
        },
        "visual_change_scores": visual_change_scores,
        "frames": frame_records,
        "transcript": transcript,
        "evidence_path": str(evidence_path),
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_task_without_video(source_task_dir: Path, destination_task_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for source in sorted(source_task_dir.rglob("*")):
        relative = source.relative_to(source_task_dir)
        destination = destination_task_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(source)
            continue
        _link_or_copy(source, destination)
    return videos


def _copy_derived_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    *,
    expected_evidence_mode: str,
    expected_attach_keyframes: bool,
) -> None:
    evidence_path = source_task_dir / "context" / "video_evidence.md"
    if not evidence_path.is_file():
        raise ValueError(f"Reusable task is missing generated video evidence: {source_task_dir}")
    manifest_path = source_task_dir / "context" / "video_evidence" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Reusable task is missing video manifest: {source_task_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_mode = str(manifest.get("evidence_mode", "ocr"))
    if actual_mode != expected_evidence_mode:
        raise ValueError(
            f"Reusable task mode is {actual_mode!r}, expected "
            f"{expected_evidence_mode!r}: {source_task_dir}"
        )
    actual_attach_keyframes = bool(manifest.get("attach_keyframes", False))
    if actual_attach_keyframes != expected_attach_keyframes:
        raise ValueError(
            f"Reusable task attach_keyframes is {actual_attach_keyframes!r}, expected "
            f"{expected_attach_keyframes!r}: {source_task_dir}"
        )
    for source in sorted(source_task_dir.rglob("*")):
        relative = source.relative_to(source_task_dir)
        destination = destination_task_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            _link_or_copy(source, destination)


def materialize_video_ocr_dataset(
    *,
    dataset_root: Path,
    gold_root: Path,
    suite_path: Path,
    output_root: Path,
    interval_seconds: float = 10.0,
    max_frames: int = 12,
    minimum_confidence: float = 0.45,
    evidence_mode: str = "ocr",
    scene_probe_interval_seconds: float = 1.0,
    scene_change_threshold: float = 0.12,
    scene_min_gap_seconds: float = 2.0,
    asr_model_name: str = "small",
    asr_language: str | None = None,
    asr_device: str = "cpu",
    asr_compute_type: str = "int8",
    attach_keyframes: bool = True,
    reuse_input_root: Path | None = None,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> tuple[Path, Path, Path]:
    if evidence_mode not in VIDEO_EVIDENCE_MODES:
        raise ValueError(
            f"Unsupported evidence_mode {evidence_mode!r}; "
            f"choose one of {sorted(VIDEO_EVIDENCE_MODES)}."
        )
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}. "
            "Choose a new directory or remove the incomplete output explicitly."
        )
    dataset = DABenchPublicDataset(dataset_root)
    task_ids = load_suite_task_ids(suite_path)
    input_root = output_root / "input"
    output_gold_root = output_root / "output"
    input_root.mkdir(parents=True)
    output_gold_root.mkdir(parents=True)

    _, RapidOCR = _video_dependencies()
    ocr_engine = RapidOCR()
    asr_model = None
    if evidence_mode == "ocr_keyframes_asr":
        WhisperModel = _asr_dependency()
        asr_model = WhisperModel(
            asr_model_name,
            device=asr_device,
            compute_type=asr_compute_type,
        )
    task_manifests: list[dict[str, Any]] = []
    total_tasks = len(task_ids)
    for task_index, task_id in enumerate(task_ids, start=1):
        task = dataset.get_task(task_id)
        destination_task_dir = input_root / task_id
        reusable_task_dir = reuse_input_root / task_id if reuse_input_root is not None else None
        reused = reusable_task_dir is not None and reusable_task_dir.is_dir()
        if reused:
            _copy_derived_task(
                reusable_task_dir,
                destination_task_dir,
                expected_evidence_mode=evidence_mode,
                expected_attach_keyframes=evidence_mode != "ocr" and attach_keyframes,
            )
            video_manifests: list[dict[str, Any]] = []
        else:
            videos = _copy_task_without_video(task.task_dir, destination_task_dir)
            if not videos:
                raise ValueError(f"Suite task does not contain a video: {task_id}")
            context_dir = destination_task_dir / "context"
            video_manifests = [
                preprocess_video(
                    video_path=video_path,
                    output_context_dir=context_dir,
                    ocr_engine=ocr_engine,
                    interval_seconds=interval_seconds,
                    max_frames=max_frames,
                    minimum_confidence=minimum_confidence,
                    evidence_mode=evidence_mode,
                    scene_probe_interval_seconds=scene_probe_interval_seconds,
                    scene_change_threshold=scene_change_threshold,
                    scene_min_gap_seconds=scene_min_gap_seconds,
                    asr_model=asr_model,
                    asr_language=asr_language,
                    attach_keyframes=attach_keyframes,
                )
                for video_path in videos
            ]
        source_gold_dir = gold_root / task_id
        if source_gold_dir.is_dir():
            for source in source_gold_dir.rglob("*"):
                if source.is_file():
                    _link_or_copy(
                        source,
                        output_gold_root / task_id / source.relative_to(source_gold_dir),
                    )
        task_manifests.append(
            {
                "task_id": task_id,
                "source_task_dir": str(task.task_dir),
                "videos": video_manifests,
                "reused_from": str(reusable_task_dir) if reused else None,
            }
        )
        if progress_callback is not None:
            progress_callback(task_index, total_tasks, task_id, reused)

    manifest = {
        "mode": evidence_mode,
        "dataset_root": str(dataset_root),
        "gold_root": str(gold_root),
        "suite_path": str(suite_path),
        "preprocessing": {
            "evidence_mode": evidence_mode,
            "interval_seconds": interval_seconds,
            "max_frames": max_frames,
            "minimum_ocr_confidence": minimum_confidence,
            "scene_probe_interval_seconds": scene_probe_interval_seconds,
            "scene_change_threshold": scene_change_threshold,
            "scene_min_gap_seconds": scene_min_gap_seconds,
            "asr_model_name": (asr_model_name if evidence_mode == "ocr_keyframes_asr" else None),
            "asr_language": asr_language,
            "asr_device": asr_device if evidence_mode == "ocr_keyframes_asr" else None,
            "asr_compute_type": (
                asr_compute_type if evidence_mode == "ocr_keyframes_asr" else None
            ),
            "attach_keyframes": evidence_mode != "ocr" and attach_keyframes,
        },
        "task_count": len(task_manifests),
        "input_root": str(input_root),
        "output_root": str(output_gold_root),
        "tasks": task_manifests,
    }
    manifest_path = output_root / "video_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return input_root, output_gold_root, manifest_path
