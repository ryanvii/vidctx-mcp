"""三层渐进式视频上下文核心引擎。

将视频理解拆解为三个递进层级，按需消耗上下文预算：
1. ``peek_video_map``：只返回目录/时间切片、极简首句预览与完整转录 Token 估算；
2. ``search_video_moments``：基于双语 BM25 检索相关时间窗口，只返回高命中片段；
3. ``inspect_frame_range``：仅针对命中的窄区间（默认限制 <= 60s）精确抽帧、评分去重并对齐原话。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .backend import MediaBackend, SubprocessMediaBackend
from .cache import WorkspaceCache
from .fallback_index import build_fallback_segments
from .models import (
    ChapterSummary,
    EvidenceBundle,
    FrameEvidence,
    MediaInfo,
    MomentSearchResult,
    SubtitleSegment,
    VideoMap,
    format_timestamp,
)
from .search import search_moments
from .source import AUDIO_EXTENSIONS, parse_source
from .subtitles import estimate_tokens, slice_segments_by_range
from .vision import select_informative_frames


class VideoContextEngine:
    """编排缓存、媒体后端、检索与视觉帧去重的核心服务类。"""

    def __init__(
        self,
        cache: WorkspaceCache | None = None,
        backend: MediaBackend | None = None,
    ) -> None:
        self.cache = cache or WorkspaceCache()
        self.backend = backend or SubprocessMediaBackend()

    def _load_or_index(
        self,
        source: str,
        unmask_finance: bool = False,
        refresh: bool = False,
        fallback_index: Literal["none", "ocr", "asr", "auto"] = "none",
    ) -> tuple[MediaInfo, list[SubtitleSegment]]:
        parsed = parse_source(source)
        vdir = self.cache.video_dir(parsed.video_id)
        info: MediaInfo | None = None
        segments: list[SubtitleSegment] = []

        if not refresh:
            cached = self.cache.load_index(parsed.video_id)
            if cached is not None:
                info, segments = cached

        if info is None:
            info, segments = self.backend.inspect_and_load_subtitles(
                parsed,
                work_dir=vdir,
                unmask_finance=unmask_finance,
            )

        if not segments and fallback_index != "none" and info.source_type != "local_subtitle":
            media_path = self.backend.ensure_media_file(info, vdir)
            segments, sub_source = build_fallback_segments(
                mode=fallback_index,
                media_path=media_path,
                work_dir=vdir,
                duration_sec=info.duration_sec,
                unmask_finance=unmask_finance,
            )
            if segments:
                info = info.model_copy(update={"subtitle_source": sub_source})

        self.cache.save_index(parsed.video_id, info, segments)
        return info, segments

    def peek_video_map(
        self,
        source: str,
        max_chapters: int = 12,
        max_preview_chars: int = 72,
        unmask_finance: bool = False,
        refresh: bool = False,
        fallback_index: Literal["none", "ocr", "asr", "auto"] = "none",
    ) -> VideoMap:
        """Layer 1：极低 Token 成本返回视频结构地图，不解码画面像素。"""
        info, segments = self._load_or_index(
            source,
            unmask_finance=unmask_finance,
            refresh=refresh,
            fallback_index=fallback_index,
        )
        chapters = self._build_chapters(
            info=info,
            segments=segments,
            max_chapters=max(1, max_chapters),
            max_preview_chars=max(20, max_preview_chars),
        )
        full_text = "\n".join(seg.text for seg in segments)
        full_tokens = estimate_tokens(full_text)

        if segments:
            hint = (
                "Call `search_video_moments` with specific keywords to locate candidate intervals, "
                "then call `inspect_frame_range` on the target 10-20s window."
            )
        else:
            hint = (
                "No text captions found. Retry with `fallback_index='ocr'` (or `'asr'`/`'auto'`) "
                "to build a cached transcript index, OR call `inspect_frame_range` on key intervals directly."
            )

        draft = VideoMap(
            video_id=info.video_id,
            title=info.title,
            duration_sec=round(info.duration_sec, 2),
            duration_ts=format_timestamp(info.duration_sec),
            source_type=info.source_type,
            has_subtitles=bool(segments),
            subtitle_source=info.subtitle_source,
            segment_count=len(segments),
            estimated_full_tokens=full_tokens,
            map_tokens=0,
            chapters=chapters,
            next_step_hint=hint,
        )
        draft.map_tokens = estimate_tokens(json.dumps(draft.model_dump(), ensure_ascii=False))
        return draft

    def search_video_moments(
        self,
        source: str,
        query: str,
        top_k: int = 5,
        window_sec: float = 20.0,
        unmask_finance: bool = False,
        fallback_index: Literal["none", "ocr", "asr", "auto"] = "none",
    ) -> MomentSearchResult:
        """Layer 2：按查询词检索字幕，返回合并后的高精度候选时间段。"""
        info, segments = self._load_or_index(
            source,
            unmask_finance=unmask_finance,
            fallback_index=fallback_index,
        )
        if not segments:
            return MomentSearchResult(
                video_id=info.video_id,
                query=query,
                total_segments_searched=0,
                hits=[],
                note="No subtitle/transcript index available; fallback to `inspect_frame_range` sampling.",
            )

        hits = search_moments(
            segments=segments,
            query=query,
            duration_sec=info.duration_sec,
            top_k=max(1, top_k),
            window_sec=max(4.0, window_sec),
        )
        note = "" if hits else "No strong keyword match found; try broader terms or inspect chapter windows."
        return MomentSearchResult(
            video_id=info.video_id,
            query=query,
            total_segments_searched=len(segments),
            hits=hits,
            note=note,
        )

    def inspect_frame_range(
        self,
        source: str,
        start_sec: float,
        end_sec: float,
        max_frames: int = 4,
        max_window_sec: float = 60.0,
        unmask_finance: bool = False,
    ) -> EvidenceBundle:
        """Layer 3：仅在指定窄区间内抽帧、评分去重，并与精确原话对齐。"""
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        requested_start = max(0.0, float(start_sec))
        requested_end = max(requested_start + 0.5, float(end_sec))

        info, segments = self._load_or_index(source, unmask_finance=unmask_finance)
        warnings: list[str] = []

        actual_start = requested_start
        actual_end = requested_end
        if info.duration_sec > 0:
            actual_start = min(actual_start, max(0.0, info.duration_sec - 0.5))
            actual_end = min(actual_end, info.duration_sec)
        if actual_end - actual_start > max_window_sec:
            actual_end = actual_start + max_window_sec
            warnings.append(
                f"Window clamped to {max_window_sec:.0f}s to protect agent context budget."
            )

        transcript_lines = slice_segments_by_range(segments, actual_start, actual_end)

        frame_evidences: list[FrameEvidence] = []
        media_suffix = Path(info.local_media_path or info.canonical_source).suffix.lower()
        if info.source_type == "local_subtitle":
            warnings.append("Subtitle-only source: visual frames are not available.")
        elif media_suffix in AUDIO_EXTENSIONS:
            warnings.append("Audio-only source: visual frames are not available.")
        else:
            vdir = self.cache.video_dir(info.video_id)
            fdir = self.cache.frames_dir(info.video_id, actual_start, actual_end)
            sample_count = max(max_frames * 2, 6)
            candidates = self.backend.extract_frames_in_range(
                info=info,
                work_dir=vdir,
                out_dir=fdir,
                start_sec=actual_start,
                end_sec=actual_end,
                sample_count=sample_count,
            )
            selected = select_informative_frames(candidates, max_frames=max(1, max_frames))
            for item in selected:
                frame_evidences.append(
                    FrameEvidence(
                        timestamp_sec=round(item.timestamp_sec, 2),
                        timestamp_ts=format_timestamp(item.timestamp_sec),
                        image_path=str(item.path),
                        info_score=item.info_score,
                        dhash=item.dhash,
                        nearest_subtitle=self._nearest_subtitle_text(item.timestamp_sec, transcript_lines),
                    )
                )

        return EvidenceBundle(
            video_id=info.video_id,
            requested_start_sec=round(requested_start, 2),
            requested_end_sec=round(requested_end, 2),
            actual_start_sec=round(actual_start, 2),
            actual_end_sec=round(actual_end, 2),
            window_ts=f"{format_timestamp(actual_start)} - {format_timestamp(actual_end)}",
            transcript_lines=transcript_lines,
            frames=frame_evidences,
            warnings=warnings,
        )

    @staticmethod
    def _nearest_subtitle_text(ts: float, lines: list[SubtitleSegment]) -> str:
        if not lines:
            return ""
        best = min(lines, key=lambda seg: abs(((seg.start + seg.end) / 2.0) - ts))
        return best.text

    @staticmethod
    def _build_chapters(
        info: MediaInfo,
        segments: list[SubtitleSegment],
        max_chapters: int,
        max_preview_chars: int,
    ) -> list[ChapterSummary]:
        duration = info.duration_sec or (segments[-1].end if segments else 0.0)

        # 优先使用视频原生章节信息
        if info.chapters:
            summaries: list[ChapterSummary] = []
            for idx, ch in enumerate(info.chapters[:max_chapters], start=1):
                start = float(ch.get("start_time", 0.0))
                end = float(ch.get("end_time", start))
                title = str(ch.get("title") or f"Section {idx}").strip()
                in_range = slice_segments_by_range(segments, start, end)
                preview = " ".join(s.text for s in in_range[:2]).strip()
                if len(preview) > max_preview_chars:
                    preview = preview[: max_preview_chars - 3] + "..."
                summaries.append(
                    ChapterSummary(
                        index=idx,
                        title=title,
                        start_sec=round(start, 2),
                        end_sec=round(end, 2),
                        start_ts=format_timestamp(start),
                        end_ts=format_timestamp(end),
                        preview=preview,
                    )
                )
            return summaries

        # 无原生章节时，按总时长均匀切分为导航桶，每个桶只取代表性首句作为超轻预览
        if duration <= 0.0:
            return []
        bucket_count = min(max_chapters, max(1, len(segments) // 3) if segments else min(6, max_chapters))
        bucket_span = duration / bucket_count
        summaries = []
        for idx in range(bucket_count):
            start = idx * bucket_span
            end = min(duration, (idx + 1) * bucket_span)
            in_range = slice_segments_by_range(segments, start, end)
            preview = " ".join(s.text for s in in_range[:2]).strip() if in_range else "(no captions in window)"
            if len(preview) > max_preview_chars:
                preview = preview[: max_preview_chars - 3] + "..."
            summaries.append(
                ChapterSummary(
                    index=idx + 1,
                    title=f"Segment {idx + 1}",
                    start_sec=round(start, 2),
                    end_sec=round(end, 2),
                    start_ts=format_timestamp(start),
                    end_ts=format_timestamp(end),
                    preview=preview,
                )
            )
        return summaries
