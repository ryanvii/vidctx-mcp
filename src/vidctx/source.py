"""输入源解析与安全边界校验模块。

设计原则：
1. 输入源严格限定为本地媒体/字幕文件，或通过成熟上游（yt-dlp）处理的公开稳定 URL；
2. 显式剥离跟踪参数并生成稳定的 ``video_id`` 哈希，避免同一视频因分享参数不同而重复缓存；
3. 不在核心仓库内夹带任何高风控平台的网页逆向或 Cookie 爬虫逻辑。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, unquote, urlencode, urlparse

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v", ".ts",
} | AUDIO_EXTENSIONS
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".json"}

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "spm_id_from", "from_spmid", "vd_source", "share_source", "share_medium",
    "share_plat", "share_session_id", "timestamp", "unique_k", "bbid",
    "ts", "si", "feature", "pp", "t",
}


class UnsupportedSourceError(ValueError):
    """输入源无效、不存在或超出支持边界时抛出。"""


@dataclass(frozen=True)
class ParsedSource:
    """规范化后的输入源描述对象。"""

    raw_source: str
    canonical_source: str
    source_type: Literal["local_video", "local_subtitle", "remote_url"]
    video_id: str
    local_path: Path | None = None
    sidecar_subtitle: Path | None = None


def clean_remote_url(url: str) -> str:
    """移除 URL 中的常见跟踪参数，保留决定视频内容的查询字段（如 ``v``、``p``）。"""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise UnsupportedSourceError(f"Unsupported URL scheme or missing host: {url}")

    qs = parse_qs(parsed.query, keep_blank_values=False)
    cleaned = {
        key: values[0]
        for key, values in sorted(qs.items())
        if key.lower() not in _TRACKING_PARAMS and values
    }
    query = urlencode(cleaned)
    base = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
    return f"{base}?{query}" if query else base


def _find_sidecar_subtitle(media_path: Path) -> Path | None:
    """查找与本地视频同名的旁挂字幕文件（优先级：.srt > .vtt > .json）。"""
    for ext in (".srt", ".vtt", ".json"):
        candidate = media_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def parse_source(source: str) -> ParsedSource:
    """将本地路径、``file://`` URI 或 HTTP(S) URL 规范化为 ``ParsedSource``。"""
    text = (source or "").strip().strip('"').strip("'")
    if not text:
        raise UnsupportedSourceError("Source must not be empty.")

    if text.startswith("file://"):
        text = unquote(urlparse(text).path)

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        canonical = clean_remote_url(text)
        return ParsedSource(
            raw_source=source,
            canonical_source=canonical,
            source_type="remote_url",
            video_id=f"url_{_hash_id(canonical)}",
        )

    expanded = Path(os.path.expanduser(text)).resolve()
    if not expanded.exists():
        raise UnsupportedSourceError(f"Local file does not exist: {expanded}")
    if expanded.is_dir():
        raise UnsupportedSourceError(f"Expected a file path, got directory: {expanded}")

    suffix = expanded.suffix.lower()
    if suffix in SUBTITLE_EXTENSIONS:
        return ParsedSource(
            raw_source=source,
            canonical_source=str(expanded),
            source_type="local_subtitle",
            video_id=f"sub_{_hash_id(str(expanded))}",
            local_path=expanded,
            sidecar_subtitle=expanded,
        )

    if suffix in VIDEO_EXTENSIONS or suffix == "":
        return ParsedSource(
            raw_source=source,
            canonical_source=str(expanded),
            source_type="local_video",
            video_id=f"vid_{_hash_id(str(expanded))}",
            local_path=expanded,
            sidecar_subtitle=_find_sidecar_subtitle(expanded),
        )

    raise UnsupportedSourceError(
        f"Unsupported file extension '{suffix}'. Supported: {sorted(VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS)}"
    )
