"""端到端三层渐进式工作流测试（含 FakeMediaBackend 注入验证按需抽帧）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from vidctx.cache import WorkspaceCache
from vidctx.engine import VideoContextEngine
from vidctx.models import MediaInfo, SubtitleSegment
from vidctx.source import ParsedSource


class FakeMediaBackend:
    """确定性测试后端：记录抽帧调用次数并生成真实 Pillow 图像。"""

    def __init__(self) -> None:
        self.inspect_calls = 0
        self.extract_calls = 0
        self.last_range: tuple[float, float] | None = None

    def ensure_media_file(self, info: MediaInfo, work_dir: Path) -> Path:
        assert info.local_media_path is not None
        return Path(info.local_media_path)

    def inspect_and_load_subtitles(
        self,
        parsed: ParsedSource,
        work_dir: Path,
        unmask_finance: bool = False,
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        self.inspect_calls += 1
        segments = [
            SubtitleSegment(start=0.0, end=15.0, text="开场介绍 Transformer 整体结构。"),
            SubtitleSegment(start=120.0, end=135.0, text="讲解 Attention QKV 矩阵乘法细节。"),
            SubtitleSegment(start=610.0, end=620.0, text="现在来看 LayerNorm 与 RMSNorm 的核心差异。"),
            SubtitleSegment(start=621.0, end=630.0, text="RMSNorm 去掉了减均值操作，计算开销显著下降。"),
            SubtitleSegment(start=1750.0, end=1790.0, text="课程总结与参考资料链接。"),
        ]
        info = MediaInfo(
            video_id=parsed.video_id,
            title="MiniMind 第7集：RMSNorm 原理",
            duration_sec=1800.0,
            source_type=parsed.source_type,
            canonical_source=parsed.canonical_source,
            chapters=[
                {"title": "课程导入", "start_time": 0.0, "end_time": 120.0},
                {"title": "Attention 回顾", "start_time": 120.0, "end_time": 600.0},
                {"title": "RMSNorm 核心推导", "start_time": 600.0, "end_time": 1200.0},
                {"title": "总结", "start_time": 1200.0, "end_time": 1800.0},
            ],
            local_media_path=str(parsed.local_path) if parsed.local_path else None,
            subtitle_source="sidecar_srt",
        )
        return info, segments

    def extract_frames_in_range(
        self,
        info: MediaInfo,
        work_dir: Path,
        out_dir: Path,
        start_sec: float,
        end_sec: float,
        sample_count: int = 8,
    ) -> list[tuple[float, Path]]:
        self.extract_calls += 1
        self.last_range = (start_sec, end_sec)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 在窄区间内制造：2 张重复幻灯片 + 1 张空白帧 + 1 张新公式页
        results: list[tuple[float, Path]] = []
        specs = [
            (start_sec + 1.0, "blank"),
            (start_sec + 3.0, "slide_a"),
            (start_sec + 5.0, "slide_a"),
            (start_sec + 8.0, "slide_b"),
        ]
        for idx, (ts, kind) in enumerate(specs):
            p = out_dir / f"fake_{idx}.jpg"
            img = Image.new("RGB", (480, 270), color=(10, 10, 10) if kind == "blank" else (245, 248, 252))
            draw = ImageDraw.Draw(img)
            if kind == "slide_a":
                draw.rectangle([20, 20, 460, 70], fill=(37, 99, 235))
                draw.line([30, 120, 440, 120], fill=(15, 23, 42), width=4)
            elif kind == "slide_b":
                draw.ellipse([60, 50, 220, 210], outline=(185, 28, 28), width=5)
                draw.polygon([(280, 220), (360, 60), (440, 220)], fill=(217, 119, 6))
            img.save(p)
            results.append((ts, p))
        return results


def test_progressive_three_layer_workflow(tmp_path: Path) -> None:
    video_file = tmp_path / "lecture.mp4"
    video_file.write_bytes(b"dummy-video")

    backend = FakeMediaBackend()
    cache = WorkspaceCache(root_dir=tmp_path / "cache")
    engine = VideoContextEngine(cache=cache, backend=backend)

    # Layer 1: 只拿目录结构与 Token 预算，绝对不抽帧
    vmap = engine.peek_video_map(str(video_file))
    assert vmap.duration_sec == 1800.0
    assert len(vmap.chapters) == 4
    assert backend.extract_calls == 0

    # Layer 2: 检索命中的是第 600s 附近的 RMSNorm 区间，且命中索引缓存，依然不抽帧
    search_res = engine.search_video_moments(str(video_file), "RMSNorm 核心差异")
    assert len(search_res.hits) >= 1
    hit = search_res.hits[0]
    assert 600.0 <= hit.start_sec <= 625.0
    assert backend.inspect_calls == 1
    assert backend.extract_calls == 0

    # Layer 3: 只对命中窗口抽帧，且超大窗口会被自动截断保护
    bundle = engine.inspect_frame_range(
        str(video_file),
        start_sec=hit.start_sec,
        end_sec=hit.start_sec + 120.0,
        max_frames=3,
        max_window_sec=20.0,
    )
    assert backend.extract_calls == 1
    assert bundle.actual_end_sec - bundle.actual_start_sec == 20.0
    assert any("clamped" in w for w in bundle.warnings)
    assert len(bundle.transcript_lines) >= 1
    # 2 张重复 slide_a 应去重为 1 张；空白帧若保留也排在后面或被高分帧替代
    assert len(bundle.frames) <= 3
    assert all(Path(f.image_path).is_file() for f in bundle.frames)
    assert any(f.nearest_subtitle for f in bundle.frames)


def test_inspect_frame_range_skips_frames_for_audio_only_source(tmp_path: Path) -> None:
    audio_file = tmp_path / "podcast.mp3"
    audio_file.write_bytes(b"fake-mp3")

    backend = FakeMediaBackend()
    engine = VideoContextEngine(
        cache=WorkspaceCache(root_dir=tmp_path / "audio_cache"),
        backend=backend,
    )
    bundle = engine.inspect_frame_range(str(audio_file), start_sec=0.0, end_sec=16.0)
    assert backend.extract_calls == 0
    assert bundle.frames == []
    assert any("Audio-only" in w for w in bundle.warnings)
