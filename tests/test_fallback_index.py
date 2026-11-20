"""Cookie 透传参数与无字幕视频 OCR/ASR 回退索引测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from vidctx.backend import SubprocessMediaBackend
from vidctx.cache import WorkspaceCache
from vidctx.engine import VideoContextEngine
from vidctx.fallback_index import merge_ocr_samples
from vidctx.models import MediaInfo, SubtitleSegment
from vidctx.source import ParsedSource


class _SubtitlelessBackend:
    """模拟一个完全没有官方/旁挂字幕的本地视频后端。"""

    def inspect_and_load_subtitles(
        self,
        parsed: ParsedSource,
        work_dir: Path,
        unmask_finance: bool = False,
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        assert parsed.local_path is not None
        return (
            MediaInfo(
                video_id=parsed.video_id,
                title="硬字幕测试视频",
                duration_sec=30.0,
                source_type="local_video",
                canonical_source=parsed.canonical_source,
                local_media_path=str(parsed.local_path),
                subtitle_source="none",
            ),
            [],
        )

    def ensure_media_file(self, info: MediaInfo, work_dir: Path) -> Path:
        assert info.local_media_path is not None
        return Path(info.local_media_path)

    def extract_frames_in_range(
        self,
        info: MediaInfo,
        work_dir: Path,
        out_dir: Path,
        start_sec: float,
        end_sec: float,
        sample_count: int = 8,
    ) -> list[tuple[float, Path]]:
        return []


def test_ytdlp_cookie_args_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_file = tmp_path / "bili_cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    monkeypatch.setenv("VIDCTX_COOKIES_FILE", str(cookie_file))
    backend = SubprocessMediaBackend()
    assert backend._ytdlp_cookie_args() == ["--cookies", str(cookie_file)]

    monkeypatch.delenv("VIDCTX_COOKIES_FILE", raising=False)
    monkeypatch.setenv("VIDCTX_COOKIES_FROM_BROWSER", "chrome")
    assert backend._ytdlp_cookie_args() == ["--cookies-from-browser", "chrome"]


def test_merge_ocr_samples_collapses_duplicate_captions() -> None:
    samples = [
        (0.0, ""),
        (4.0, "不是随便超过前面一根线"),
        (8.0, "不是随便超过前面一根线"),
        (12.0, "   "),
        (16.0, "不是随便超过前面一根线"),
        (20.0, "就会画出结构突破 BOS"),
        (24.0, "就会画出结构突破 BOS"),
    ]
    segments = merge_ocr_samples(samples, interval_sec=4.0, duration_sec=30.0)
    assert len(segments) == 2
    assert segments[0].start == 4.0
    assert segments[0].end == 20.0
    assert "前面一根线" in segments[0].text
    assert segments[1].start == 20.0
    assert "BOS" in segments[1].text


def test_engine_fallback_index_populates_and_caches_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "smc_hardsub.mp4"
    video.write_bytes(b"fake-video")

    calls = 0

    def _fake_build_fallback(**kwargs):
        nonlocal calls
        calls += 1
        return (
            [
                SubtitleSegment(start=12.0, end=18.0, text="结构突破 BOS 必须收线实体确认"),
                SubtitleSegment(start=20.0, end=28.0, text="CHoCH 特征改变意味着趋势反转信号"),
            ],
            "fallback_ocr",
        )

    monkeypatch.setattr("vidctx.engine.build_fallback_segments", _fake_build_fallback)

    engine = VideoContextEngine(
        cache=WorkspaceCache(root_dir=tmp_path / "cache"),
        backend=_SubtitlelessBackend(),
    )

    # 默认不触发回退建索引
    vmap_plain = engine.peek_video_map(str(video))
    assert vmap_plain.has_subtitles is False
    assert calls == 0

    # 显式触发 fallback_index="ocr" 后，建索引并缓存
    vmap_ocr = engine.peek_video_map(str(video), fallback_index="ocr")
    assert vmap_ocr.has_subtitles is True
    assert vmap_ocr.subtitle_source == "fallback_ocr"
    assert calls == 1

    # 后续 search 直接命中缓存，不再重复建索引
    hits = engine.search_video_moments(str(video), "结构突破 BOS")
    assert len(hits.hits) == 1
    assert hits.hits[0].start_sec <= 15.0 <= hits.hits[0].end_sec
    assert calls == 1
