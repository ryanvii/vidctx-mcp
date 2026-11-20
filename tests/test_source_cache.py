"""输入源安全边界校验与缓存 LRU 淘汰测试。"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from vidctx.backend import MediaBackendError, SubprocessMediaBackend
from vidctx.cache import WorkspaceCache
from vidctx.models import MediaInfo, SubtitleSegment
from vidctx.source import UnsupportedSourceError, clean_remote_url, parse_source


def test_clean_remote_url_strips_tracking_params() -> None:
    url = "https://www.bilibili.com/video/BV1xx411c7mD?p=3&t=125&spm_id_from=333.999&vd_source=abc"
    cleaned = clean_remote_url(url)
    assert cleaned == "https://www.bilibili.com/video/BV1xx411c7mD?p=3"


def test_parse_source_detects_sidecar_and_rejects_invalid(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake-mp4")
    srt = tmp_path / "lesson.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")

    parsed = parse_source(str(video))
    assert parsed.source_type == "local_video"
    assert parsed.sidecar_subtitle == srt

    with pytest.raises(UnsupportedSourceError):
        parse_source(str(tmp_path / "missing.mp4"))

    with pytest.raises(UnsupportedSourceError):
        parse_source("ftp://example.com/video.mp4")

    bad_ext = tmp_path / "notes.docx"
    bad_ext.write_bytes(b"abc")
    with pytest.raises(UnsupportedSourceError):
        parse_source(str(bad_ext))


def test_workspace_cache_lru_eviction(tmp_path: Path) -> None:
    cache = WorkspaceCache(root_dir=tmp_path / "cache", max_bytes=10 * 1024 * 1024)
    # 构造两个大文件触发 LRU 淘汰（手动覆盖 max_bytes 用于测试阈值）
    cache.max_bytes = 1500

    info_old = MediaInfo(
        video_id="vid_old",
        title="old",
        duration_sec=10.0,
        source_type="local_subtitle",
        canonical_source="/tmp/old.srt",
    )
    info_new = MediaInfo(
        video_id="vid_new",
        title="new",
        duration_sec=10.0,
        source_type="local_subtitle",
        canonical_source="/tmp/new.srt",
    )
    segs = [SubtitleSegment(start=0.0, end=5.0, text="x" * 900)]

    cache.save_index("vid_old", info_old, segs)
    old_marker = cache.video_dir("vid_old") / ".last_used"
    old_time = time.time() - 3600
    os.utime(old_marker, (old_time, old_time))

    cache.save_index("vid_new", info_new, segs)
    assert not (cache.root_dir / "vid_old").exists()
    assert cache.load_index("vid_new") is not None


def test_subprocess_backend_missing_binaries_fail_gracefully(tmp_path: Path) -> None:
    backend = SubprocessMediaBackend(
        ffmpeg_bin="nonexistent_ffmpeg_bin",
        ffprobe_bin="nonexistent_ffprobe_bin",
        ytdlp_bin="nonexistent_ytdlp_bin",
    )

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "demo.srt"
    srt.write_text("1\n00:00:02,000 --> 00:00:09,500\n侧挂字幕依然可用\n", encoding="utf-8")

    parsed_local = parse_source(str(video))
    info, segs = backend.inspect_and_load_subtitles(parsed_local, tmp_path / "w1")
    assert info.duration_sec == 9.5
    assert info.subtitle_source == "sidecar_srt"
    assert len(segs) == 1

    with pytest.raises(MediaBackendError, match="Missing required binary"):
        backend.extract_frames_in_range(info, tmp_path / "w1", tmp_path / "f1", 0.0, 5.0)

    parsed_url = parse_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with pytest.raises(MediaBackendError, match="Missing required binary"):
        backend.inspect_and_load_subtitles(parsed_url, tmp_path / "w2")
