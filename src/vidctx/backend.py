"""媒体底层适配层：将外部工具（ffprobe / ffmpeg / yt-dlp）与核心检索逻辑彻底解耦。

设计要点：
- 通过 ``MediaBackend`` Protocol 抽象底层媒体操作，使单元测试无需联网或依赖真实二进制；
- 所有子进程调用统一采用 ``subprocess.run(..., timeout=...)``，杜绝 ``signal.SIGALRM`` 在异步线程或 Windows 下崩溃的问题；
- 在 Layer 1 / Layer 2 阶段通过 ``yt-dlp --skip-download`` 仅拉取元信息与字幕，真正调用 Layer 3 抽帧时才按需下载媒体文件。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from .models import MediaInfo, SubtitleSegment
from .source import ParsedSource
from .subtitles import load_subtitles_from_file, restore_masked_terms


class MediaBackendError(RuntimeError):
    """底层媒体探测、字幕提取或抽帧失败时抛出。"""


class MediaBackend(Protocol):
    """媒体后端抽象协议。"""

    def inspect_and_load_subtitles(
        self,
        parsed: ParsedSource,
        work_dir: Path,
        unmask_finance: bool = False,
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        """探测基础元信息并提取字幕（不解码视频画面像素）。"""

    def extract_frames_in_range(
        self,
        info: MediaInfo,
        work_dir: Path,
        out_dir: Path,
        start_sec: float,
        end_sec: float,
        sample_count: int = 8,
    ) -> list[tuple[float, Path]]:
        """仅针对 ``[start_sec, end_sec]`` 窄时间窗精确抽取候选帧。"""

    def ensure_media_file(self, info: MediaInfo, work_dir: Path) -> Path:
        """确保目标视频已在本地可用（供按需抽帧或可选回退建索引使用）。"""


class SubprocessMediaBackend:
    """基于系统 ``ffprobe`` / ``ffmpeg`` 与可选 ``yt-dlp`` 的生产环境后端。"""

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        ytdlp_bin: str = "yt-dlp",
        timeout_sec: int = 45,
        cookies_file: str | None = None,
        cookies_from_browser: str | None = None,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.ytdlp_bin = ytdlp_bin
        self.timeout_sec = timeout_sec
        self.cookies_file = cookies_file
        self.cookies_from_browser = cookies_from_browser

    def _ytdlp_cookie_args(self) -> list[str]:
        """读取显式参数或环境变量中的 Cookie 配置并安全透传给 yt-dlp。"""
        cfile = self.cookies_file or os.environ.get("VIDCTX_COOKIES_FILE", "").strip()
        if cfile and Path(cfile).expanduser().is_file():
            return ["--cookies", str(Path(cfile).expanduser())]
        browser = self.cookies_from_browser or os.environ.get("VIDCTX_COOKIES_FROM_BROWSER", "").strip()
        if browser:
            return ["--cookies-from-browser", browser]
        return []

    def ensure_media_file(self, info: MediaInfo, work_dir: Path) -> Path:
        return self._ensure_local_media_path(info, work_dir)

    def inspect_and_load_subtitles(
        self,
        parsed: ParsedSource,
        work_dir: Path,
        unmask_finance: bool = False,
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        if parsed.source_type == "local_subtitle":
            assert parsed.local_path is not None
            segments = load_subtitles_from_file(parsed.local_path, unmask_finance=unmask_finance)
            duration = segments[-1].end if segments else 0.0
            info = MediaInfo(
                video_id=parsed.video_id,
                title=parsed.local_path.stem,
                duration_sec=duration,
                source_type="local_subtitle",
                canonical_source=parsed.canonical_source,
                subtitle_source="local_file" if segments else "none",
            )
            return info, segments

        if parsed.source_type == "local_video":
            assert parsed.local_path is not None
            duration, chapters = self._probe_local_video(parsed.local_path)
            segments: list[SubtitleSegment] = []
            subtitle_source = "none"

            if parsed.sidecar_subtitle and parsed.sidecar_subtitle.is_file():
                segments = load_subtitles_from_file(
                    parsed.sidecar_subtitle,
                    unmask_finance=unmask_finance,
                )
                if segments:
                    subtitle_source = f"sidecar_{parsed.sidecar_subtitle.suffix.lstrip('.')}"

            if not segments:
                extracted = self._extract_embedded_subtitle(parsed.local_path, work_dir)
                if extracted is not None:
                    segments = load_subtitles_from_file(extracted, unmask_finance=unmask_finance)
                    if segments:
                        subtitle_source = "embedded_track"

            if duration <= 0.0 and segments:
                duration = segments[-1].end

            info = MediaInfo(
                video_id=parsed.video_id,
                title=parsed.local_path.stem,
                duration_sec=duration,
                source_type="local_video",
                canonical_source=parsed.canonical_source,
                chapters=chapters,
                local_media_path=str(parsed.local_path),
                subtitle_source=subtitle_source,
            )
            return info, segments

        # 远程公开链接：严格委托给 yt-dlp，且优先仅拉取元数据与字幕（--skip-download）
        return self._inspect_remote_url(parsed, work_dir, unmask_finance=unmask_finance)

    def extract_frames_in_range(
        self,
        info: MediaInfo,
        work_dir: Path,
        out_dir: Path,
        start_sec: float,
        end_sec: float,
        sample_count: int = 8,
    ) -> list[tuple[float, Path]]:
        if info.source_type == "local_subtitle":
            return []

        media_path = self._ensure_local_media_path(info, work_dir)
        self._require_binary(self.ffmpeg_bin, "Install ffmpeg to extract video frames.")

        span = max(0.5, end_sec - start_sec)
        count = max(1, sample_count)
        step = span / count if count > 1 else span
        timestamps = [
            round(min(end_sec, start_sec + step * (i + 0.5)), 3)
            for i in range(count)
        ]

        out_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[tuple[float, Path]] = []
        for idx, ts in enumerate(timestamps):
            dest = out_dir / f"frame_{idx:03d}_{ts:.2f}s.jpg"
            if dest.is_file() and dest.stat().st_size > 0:
                extracted.append((ts, dest))
                continue
            # 将 -ss 放在 -i 之前实现关键帧快速定位，避免从头解码整部长视频
            cmd = [
                self.ffmpeg_bin,
                "-ss", str(ts),
                "-i", str(media_path),
                "-frames:v", "1",
                "-q:v", "2",
                "-vf", "scale=960:-2",
                str(dest),
                "-y", "-loglevel", "error",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
            if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                extracted.append((ts, dest))
        return extracted

    def _probe_local_video(self, media_path: Path) -> tuple[float, list[dict]]:
        if not shutil.which(self.ffprobe_bin):
            return 0.0, []
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-show_chapters",
            "-of", "json",
            str(media_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
            if proc.returncode != 0 or not proc.stdout.strip():
                return 0.0, []
            data = json.loads(proc.stdout)
            duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
            chapters = [
                {
                    "title": ch.get("tags", {}).get("title", "") or f"Chapter {i + 1}",
                    "start_time": float(ch.get("start_time", 0.0) or 0.0),
                    "end_time": float(ch.get("end_time", 0.0) or 0.0),
                }
                for i, ch in enumerate(data.get("chapters", []))
            ]
            return duration, chapters
        except (subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            return 0.0, []

    def _extract_embedded_subtitle(self, media_path: Path, work_dir: Path) -> Path | None:
        if not shutil.which(self.ffmpeg_bin):
            return None
        dest = work_dir / "embedded.srt"
        cmd = [
            self.ffmpeg_bin,
            "-i", str(media_path),
            "-map", "0:s:0",
            str(dest),
            "-y", "-loglevel", "error",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
            if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                return dest
        except subprocess.SubprocessError:
            pass
        return None

    def _inspect_remote_url(
        self,
        parsed: ParsedSource,
        work_dir: Path,
        unmask_finance: bool = False,
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        self._require_binary(
            self.ytdlp_bin,
            "Remote URLs require yt-dlp. Install with `pip install 'vidctx-mcp[remote]'` or pass a local file.",
        )
        sub_dir = work_dir / "subs"
        sub_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ytdlp_bin,
            "--no-playlist",
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", "zh-Hans,zh-CN,zh,en.*",
            "--sub-format", "srt/vtt/best",
            *self._ytdlp_cookie_args(),
            "-o", str(sub_dir / "%(id)s.%(ext)s"),
            parsed.canonical_source,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise MediaBackendError(f"yt-dlp metadata fetch timed out after {self.timeout_sec}s") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[:400]
            raise MediaBackendError(
                f"yt-dlp failed to inspect URL (anti-bot or unsupported site?): {detail}"
            )

        meta = json.loads(proc.stdout)
        chapters = [
            {
                "title": ch.get("title", "") or f"Chapter {i + 1}",
                "start_time": float(ch.get("start_time", 0.0) or 0.0),
                "end_time": float(ch.get("end_time", 0.0) or 0.0),
            }
            for i, ch in enumerate(meta.get("chapters") or [])
        ]

        segments: list[SubtitleSegment] = []
        subtitle_source = "none"
        sub_files = sorted(sub_dir.glob("*.vtt")) + sorted(sub_dir.glob("*.srt"))
        if sub_files:
            segments = load_subtitles_from_file(sub_files[0], unmask_finance=unmask_finance)
            if segments:
                subtitle_source = f"ytdlp_{sub_files[0].suffix.lstrip('.')}"

        if not segments:
            yt_segments = self._try_youtube_transcript_api(
                parsed.canonical_source,
                unmask_finance=unmask_finance,
            )
            if yt_segments:
                segments = yt_segments
                subtitle_source = "youtube_transcript_api"

        info = MediaInfo(
            video_id=parsed.video_id,
            title=str(meta.get("title") or parsed.video_id),
            duration_sec=float(meta.get("duration") or (segments[-1].end if segments else 0.0)),
            source_type="remote_url",
            canonical_source=parsed.canonical_source,
            chapters=chapters,
            subtitle_source=subtitle_source,
        )
        return info, segments

    @staticmethod
    def _extract_youtube_id(url: str) -> str | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if "youtu.be" in host:
            vid = parsed.path.strip("/").split("/")[0]
            return vid if re.match(r"^[\w-]{11}$", vid) else None
        if "youtube.com" in host:
            vid = parse_qs(parsed.query).get("v", [""])[0]
            if re.match(r"^[\w-]{11}$", vid):
                return vid
            m = re.search(r"/(?:shorts|embed|live)/([\w-]{11})", parsed.path)
            return m.group(1) if m else None
        return None

    @classmethod
    def _try_youtube_transcript_api(
        cls,
        url: str,
        unmask_finance: bool = False,
    ) -> list[SubtitleSegment]:
        """当安装有 youtube-transcript-api 时，为 YouTube 链接提供轻量字幕回退。"""
        vid = cls._extract_youtube_id(url)
        if not vid:
            return []
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return []
        try:
            ytt = YouTubeTranscriptApi()
            fetched = ytt.fetch(vid, languages=["zh-Hans", "zh-CN", "zh", "en"])
            segments: list[SubtitleSegment] = []
            for item in fetched:
                text = str(getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                if unmask_finance:
                    text = restore_masked_terms(text)
                start = float(getattr(item, "start", 0.0))
                dur = float(getattr(item, "duration", 0.0))
                segments.append(
                    SubtitleSegment(start=start, end=max(start, start + dur), text=text)
                )
            return segments
        except Exception:
            return []

    def _ensure_local_media_path(self, info: MediaInfo, work_dir: Path) -> Path:
        if info.local_media_path and Path(info.local_media_path).is_file():
            return Path(info.local_media_path)

        cached_media = next(iter(sorted((work_dir / "media").glob("source.*"))), None)
        if cached_media and cached_media.is_file():
            return cached_media

        if info.source_type != "remote_url":
            raise MediaBackendError(f"Media file is unavailable for {info.video_id}")

        self._require_binary(self.ytdlp_bin, "yt-dlp is required to fetch remote video frames.")
        media_dir = work_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ytdlp_bin,
            "--no-playlist",
            "-f", "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/b",
            "--no-warnings",
            *self._ytdlp_cookie_args(),
            "-o", str(media_dir / "source.%(ext)s"),
            info.canonical_source,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(120, self.timeout_sec * 3))
        if proc.returncode != 0:
            raise MediaBackendError(f"yt-dlp media download failed: {(proc.stderr or proc.stdout).strip()[:400]}")

        downloaded = next(iter(sorted(media_dir.glob("source.*"))), None)
        if not downloaded:
            raise MediaBackendError("yt-dlp completed but no media file was produced.")
        return downloaded

    @staticmethod
    def _require_binary(binary: str, hint: str) -> None:
        if not shutil.which(binary):
            raise MediaBackendError(f"Missing required binary '{binary}'. {hint}")
