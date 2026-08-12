"""视频处理器：关键帧抽取 + ASR 转录

关键帧采用 KDD Cup Top1 方案的思路，用 pyav（av）实现：
- 只读 video stream 的编码包大小（不解码全部画面），定位字节突变的 I 帧/场景锚点
- 把近邻 burst 聚类成一组，每组取包最大的代表帧
- 只对选中的关键帧解码导出 JPG（build_frame_index）
- ASR 用 whisper（可选，缺失时优雅降级）
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path


class VideoProcessor:
    """视频 → 关键帧 + ASR 转录文本"""

    # ---------- 关键帧检测 ----------

    @staticmethod
    def extract_keyframes_by_packet_size(
        video_path: Path,
        threshold_ratio: float = 3.0,
        min_burst_gap: int = 5,
    ) -> list[tuple[int, float]]:
        """
        基于编码包大小抽取关键帧位置（Top1 思路，pyav 实现，不依赖系统 ffmpeg）。

        算法：
          1. demux video stream，记录每个 packet 的 (pts, size)（不解码画面）。
          2. 包大小 >= 滚动中位数 * threshold_ratio → 视作 burst（I 帧/场景锚点）。
          3. min_burst_gap 内的连续 burst 聚成一个簇，取包最大的为代表帧。

        返回：[(frame_index, timestamp_seconds), ...]
        """
        try:
            import av
        except ImportError:
            return []

        try:
            container = av.open(str(video_path))
        except Exception:
            return []

        stream = container.streams.video[0]
        # 必须在 close() 之前读取，否则流元数据会失效
        tbs = stream.time_base
        rate = stream.average_rate
        sizes: list[int] = []
        pts_list: list[float] = []
        for packet in container.demux(stream):
            if packet.size is None or packet.pts is None:
                continue
            sizes.append(packet.size)
            pts_list.append(packet.pts)
        container.close()

        if len(sizes) < 3:
            return []

        # 滚动中位数基线（窗口 21）
        baselines = []
        for i, s in enumerate(sizes):
            window = sizes[max(0, i - 10): i + 11]
            baselines.append(statistics.median(window))

        # burst 检测
        burst_idx = [
            i for i, s in enumerate(sizes)
            if baselines[i] > 0 and s >= threshold_ratio * baselines[i]
        ]

        # 聚类：min_burst_gap 内的连续 burst 归为一簇
        clusters: list[list[int]] = []
        for i in burst_idx:
            if clusters and i - clusters[-1][-1] <= min_burst_gap:
                clusters[-1].append(i)
            else:
                clusters.append([i])

        # 每簇取包最大的代表帧
        representatives = [max(cl, key=lambda j: sizes[j]) for cl in clusters]

        # time_base/fps 可能为 None，需防御（tbs/rate 已在 close 前捕获）
        fps = float(rate) if rate else 25.0
        out: list[tuple[int, float]] = []
        for i in representatives:
            if tbs:
                ts = float(pts_list[i] * tbs)
            else:
                ts = i / fps  # time_base 缺失时用包序号近似
            frame_idx = int(round(ts * fps))
            out.append((frame_idx, round(ts, 2)))
        return out

    @staticmethod
    def extract_keyframes_by_frame_diff(
        video_path: Path,
        threshold: float = 0.15,
        sample_every: int = 3,
    ) -> list[tuple[int, float]]:
        """备选方法：逐帧解码做相邻帧差异（均值绝对差），突变处为场景切换。

        比包大小检测慢（需解码），但对非 H.264 编码更稳。返回 [(frame_index, ts)]。
        """
        try:
            import av
        except ImportError:
            return []

        try:
            container = av.open(str(video_path))
        except Exception:
            return []
        stream = container.streams.video[0]

        prev: object = None
        keyframes: list[tuple[int, float]] = []
        for i, frame in enumerate(container.decode(stream)):
            if i % sample_every != 0:
                continue
            img = frame.to_ndarray(format="gray")
            if prev is not None:
                diff = float(abs(img.astype("float32") - prev.astype("float32")).mean() / 255.0)
                if diff > threshold:
                    keyframes.append((i, float(frame.time)))
            prev = img
            if len(keyframes) > 200:  # 上限保护
                break
        container.close()
        return keyframes

    # ---------- 帧导出 ----------

    @staticmethod
    def _decode_frame_at(video_path: Path, timestamp: float, out_path: Path) -> bool:
        """解码 timestamp 附近的最近一个关键帧并保存为 JPG。"""
        import av
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            target_pts = int(timestamp / float(stream.time_base))
            container.seek(max(0, target_pts), backward=True, stream=stream)
            for frame in container.decode(stream):
                if frame.time is not None and abs(frame.time - timestamp) > 3.0:
                    continue
                frame.to_image().convert("RGB").save(str(out_path))
                container.close()
                return True
            container.close()
            return False
        except Exception:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def build_frame_index(self, video_path: Path, output_dir: Path) -> dict:
        """
        抽关键帧 + 导出 JPG + 写索引文件。
        返回 {video, total_frames, frames: [{index, timestamp, path}, ...]}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        keyframes = self.extract_keyframes_by_packet_size(video_path)

        frame_records = []
        for idx, ts in keyframes:
            frame_path = output_dir / f"{video_path.stem}_frame_{idx:04d}.jpg"
            if self._decode_frame_at(video_path, ts, frame_path) and frame_path.exists():
                frame_records.append({"index": idx, "timestamp": ts, "path": str(frame_path)})

        index = {
            "video": str(video_path),
            "total_frames": len(frame_records),
            "frames": frame_records,
        }
        index_path = output_dir / f"{video_path.stem}_index.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return index

    # ---------- ASR ----------

    @staticmethod
    def asr_transcribe(video_path: Path) -> str:
        """ASR 转录视频音频（openai-whisper；缺失时降级）"""
        try:
            import whisper
        except ImportError:
            return f"[ASR not available for: {video_path.name}]"
        try:
            model = whisper.load_model("base")
            result = model.transcribe(str(video_path))
            return (result or {}).get("text", "")
        except Exception as e:
            return f"[ASR failed for: {video_path.name}: {e}]"

    @staticmethod
    def extract_audio(video_path: Path, output_path: Path) -> Path:
        """抽取视频音轨（优先 ffmpeg，缺失时用 pyav）"""
        import subprocess
        import sys
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 str(output_path)],
                capture_output=True,
            )
            if r.returncode == 0:
                return output_path
        except FileNotFoundError:
            pass
        # pyav 兜底
        try:
            import av
            with av.open(str(video_path)) as src:
                with av.open(str(output_path), "w") as dst:
                    for frame in src.decode(audio=0):
                        frame.pts = None
                        dst.mux(frame)
            return output_path
        except Exception as e:
            raise RuntimeError(f"无法抽取音频: {e}") from e

    # ---------- 完整处理 ----------

    def process(self, video_path: Path) -> str:
        """完整处理：关键帧 + ASR → 文本"""
        keyframes = self.extract_keyframes_by_packet_size(video_path)
        asr_text = self.asr_transcribe(video_path)

        lines = [f"VIDEO: {video_path.name}"]
        lines.append(f"Key frames: {len(keyframes)} detected")
        for idx, ts in keyframes[:10]:
            lines.append(f"  frame {idx} @ {ts:.1f}s")
        if asr_text:
            lines.append(f"ASR transcript: {asr_text[:2000]}")
        return "\n".join(lines)
