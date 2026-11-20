"""基于真实 ffmpeg/ffprobe 的端到端本地视频集成测试（未安装二进制时自动跳过）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vidctx.backend import SubprocessMediaBackend
from vidctx.cache import WorkspaceCache
from vidctx.engine import VideoContextEngine


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg and ffprobe are required for real video integration test",
)
def test_real_ffmpeg_end_to_end_pipeline(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_lecture.mp4"
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", "testsrc2=duration=6:size=640x360:rate=10",
        "-pix_fmt", "yuv420p",
        str(video_path),
        "-y", "-loglevel", "error",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert video_path.is_file()

    srt_path = tmp_path / "sample_lecture.srt"
    srt_path.write_text(
        "1\n00:00:00,500 --> 00:00:02,500\n这是课程介绍部分\n\n"
        "2\n00:00:03,000 --> 00:00:05,800\n现在展示 AgentGuard 核心评测矩阵\n",
        encoding="utf-8",
    )

    engine = VideoContextEngine(
        cache=WorkspaceCache(root_dir=tmp_path / "cache"),
        backend=SubprocessMediaBackend(),
    )

    vmap = engine.peek_video_map(str(video_path))
    assert vmap.source_type == "local_video"
    assert vmap.duration_sec >= 5.5
    assert vmap.has_subtitles is True
    assert vmap.subtitle_source == "sidecar_srt"

    hits = engine.search_video_moments(str(video_path), "AgentGuard 核心评测")
    assert len(hits.hits) >= 1

    bundle = engine.inspect_frame_range(str(video_path), start_sec=2.5, end_sec=5.8, max_frames=3)
    assert len(bundle.transcript_lines) >= 1
    assert len(bundle.frames) >= 1
    for frame in bundle.frames:
        assert Path(frame.image_path).is_file()
        assert frame.info_score > 0.2
        assert len(frame.dhash) == 16
