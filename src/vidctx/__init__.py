"""vidctx: 面向 AI Agent 的按需渐进式视频上下文读取器。"""

from .engine import VideoContextEngine
from .models import (
    ChapterSummary,
    EvidenceBundle,
    FrameEvidence,
    MediaInfo,
    MomentHit,
    MomentSearchResult,
    SubtitleSegment,
    VideoMap,
)

__version__ = "0.1.0"

__all__ = [
    "ChapterSummary",
    "EvidenceBundle",
    "FrameEvidence",
    "MediaInfo",
    "MomentHit",
    "MomentSearchResult",
    "SubtitleSegment",
    "VideoContextEngine",
    "VideoMap",
]
