"""核心数据模型定义：统一约束三层渐进式工具的输入输出结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 ``HH:MM:SS`` 字符串。"""
    clamped = max(0.0, float(seconds))
    total = int(clamped)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SubtitleSegment(BaseModel):
    """单条带时间戳的字幕或转录片段。"""

    start: float = Field(ge=0.0, description="起始时间（秒）")
    end: float = Field(ge=0.0, description="结束时间（秒）")
    text: str = Field(description="文本内容")

    @property
    def start_str(self) -> str:
        return format_timestamp(self.start)

    @property
    def end_str(self) -> str:
        return format_timestamp(self.end)


class MediaInfo(BaseModel):
    """视频基础元信息（不包含正文和帧像素）。"""

    video_id: str
    title: str
    duration_sec: float = Field(default=0.0, ge=0.0)
    source_type: Literal["local_video", "local_subtitle", "remote_url"]
    canonical_source: str
    chapters: list[dict] = Field(default_factory=list)
    local_media_path: str | None = None
    subtitle_source: str = "none"


class ChapterSummary(BaseModel):
    """视频章节或时间切片的极简摘要，用于 Layer 1 低成本导航。"""

    index: int
    title: str
    start_sec: float
    end_sec: float
    start_ts: str
    end_ts: str
    preview: str = Field(description="该时间段的精简关键词/首句预览")


class VideoMap(BaseModel):
    """Layer 1 (`peek_video_map`) 返回结构：全片目录与上下文预算估算。"""

    video_id: str
    title: str
    duration_sec: float
    duration_ts: str
    source_type: Literal["local_video", "local_subtitle", "remote_url"]
    has_subtitles: bool
    subtitle_source: str
    segment_count: int
    estimated_full_tokens: int = Field(
        description="若将完整字幕一次性注入上下文预计消耗的 Token 数"
    )
    map_tokens: int = Field(description="本次地图索引实际约占用的 Token 数")
    chapters: list[ChapterSummary]
    next_step_hint: str


class MomentHit(BaseModel):
    """Layer 2 (`search_video_moments`) 单条命中窗口。"""

    rank: int
    start_sec: float
    end_sec: float
    start_ts: str
    end_ts: str
    score: float
    matched_terms: list[str]
    snippet: str


class MomentSearchResult(BaseModel):
    """Layer 2 (`search_video_moments`) 返回结构。"""

    video_id: str
    query: str
    total_segments_searched: int
    hits: list[MomentHit]
    note: str = ""


class FrameEvidence(BaseModel):
    """Layer 3 (`inspect_frame_range`) 单张去重筛选后的高信息密度关键帧。"""

    timestamp_sec: float
    timestamp_ts: str
    image_path: str
    info_score: float = Field(ge=0.0, le=1.0)
    dhash: str
    nearest_subtitle: str = ""


class EvidenceBundle(BaseModel):
    """Layer 3 (`inspect_frame_range`) 返回结构：精确时间窗的图文证据包。"""

    video_id: str
    requested_start_sec: float
    requested_end_sec: float
    actual_start_sec: float
    actual_end_sec: float
    window_ts: str
    transcript_lines: list[SubtitleSegment]
    frames: list[FrameEvidence]
    warnings: list[str] = Field(default_factory=list)
