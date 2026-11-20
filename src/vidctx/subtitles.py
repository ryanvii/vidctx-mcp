"""多格式字幕/转录解析与上下文裁剪工具。

支持 ``.srt``、``.vtt``（WebVTT，yt-dlp 默认输出格式）以及结构化 ``.json`` 字幕片段，
并提供面向中文财经/行业短视频的可选掩码词表还原与 Token 估算。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import SubtitleSegment

_TIMESTAMP_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[,\.](\d{3})"
)
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

# 常见谐音/首字母遮罩词表（保留词边界，避免误替换 A股 / CEO / C位）
_DEFAULT_FINANCE_MASKS: tuple[tuple[str, str], ...] = (
    ("爆C", "爆仓"), ("建C", "建仓"), ("平C", "平仓"), ("补C", "补仓"),
    ("加C", "加仓"), ("减C", "减仓"), ("持C", "持仓"), ("空C", "空仓"),
    ("满C", "满仓"), ("重C", "重仓"), ("轻C", "轻仓"), ("清C", "清仓"),
    ("L润", "利润"), ("盈L", "盈利"), ("L率", "利率"), ("L息", "利息"),
    ("L好", "利好"), ("L空", "利空"), ("套L", "套利"), ("C底", "抄底"),
    ("N市", "牛市"), ("N股", "牛股"), ("X市", "熊市"), ("Q亏", "全亏"),
    ("K损", "亏损"), ("亏S", "亏损"), ("浮K", "浮亏"), ("止S", "止损"),
    ("T牢", "套牢"), ("被T", "被套"), ("解T", "解套"), ("G肉", "割肉"),
    ("G票", "股票"), ("G价", "股价"), ("G市", "股市"), ("G杆", "杠杆"),
    ("Z停", "涨停"), ("D停", "跌停"), ("Z家", "庄家"), ("J菜", "韭菜"),
    ("S户", "散户"),
)


def parse_timestamp_to_seconds(ts: str) -> float:
    """解析 ``HH:MM:SS.mmm``、``HH:MM:SS,mmm`` 或 ``MM:SS.mmm`` 为秒数。"""
    match = _TIMESTAMP_RE.search(ts.strip())
    if not match:
        raise ValueError(f"Invalid subtitle timestamp: {ts!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = int(match.group(4))
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def _clean_caption_text(raw: str) -> str:
    """移除 WebVTT 样式标签（如 ``<c>``、``<b>``）与多余空白。"""
    no_tags = re.sub(r"<[^>]+>", "", raw)
    lines = [line.strip() for line in no_tags.splitlines() if line.strip()]
    return " ".join(lines).strip()


def restore_masked_terms(text: str, extra: dict[str, str] | None = None) -> str:
    """按全词边界规则还原常见的字母替代敏感词。"""
    if not text:
        return text
    rules = dict(_DEFAULT_FINANCE_MASKS)
    if extra:
        rules.update(extra)
    for src, dst in sorted(rules.items(), key=lambda item: len(item[0]), reverse=True):
        body = "".join(
            ch if ch.isascii() and ch.isalpha() else re.escape(ch)
            for ch in src
        )
        pattern = rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"
        text = re.sub(pattern, dst, text, flags=re.IGNORECASE)
    return text


def parse_srt_text(content: str, unmask_finance: bool = False) -> list[SubtitleSegment]:
    """解析 SRT 格式文本为 ``SubtitleSegment`` 列表。"""
    normalized = content.replace("\ufeff", "").replace("\r\n", "\n").strip()
    if not normalized:
        return []

    segments: list[SubtitleSegment] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = block.strip().splitlines()
        ts_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if ts_idx is None:
            continue
        ts_parts = lines[ts_idx].split("-->", 1)
        try:
            start = parse_timestamp_to_seconds(ts_parts[0])
            end = parse_timestamp_to_seconds(ts_parts[1])
        except ValueError:
            continue
        text = _clean_caption_text("\n".join(lines[ts_idx + 1 :]))
        if not text:
            continue
        if unmask_finance:
            text = restore_masked_terms(text)
        segments.append(SubtitleSegment(start=start, end=max(start, end), text=text))
    return segments


def parse_vtt_text(content: str, unmask_finance: bool = False) -> list[SubtitleSegment]:
    """解析 WebVTT (``.vtt``) 格式字幕，自动跳过头部元数据与连续重复行。"""
    normalized = content.replace("\ufeff", "").replace("\r\n", "\n").strip()
    if not normalized:
        return []

    segments: list[SubtitleSegment] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        ts_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if ts_idx is None:
            continue
        ts_parts = lines[ts_idx].split("-->", 1)
        try:
            start = parse_timestamp_to_seconds(ts_parts[0])
            end = parse_timestamp_to_seconds(ts_parts[1].split()[0])
        except (ValueError, IndexError):
            continue
        text = _clean_caption_text("\n".join(lines[ts_idx + 1 :]))
        if not text:
            continue
        if unmask_finance:
            text = restore_masked_terms(text)
        # YouTube 自动字幕常出现前后时间块重复上一句的情况，做轻量去重
        if segments and segments[-1].text == text and abs(start - segments[-1].end) < 0.5:
            segments[-1] = SubtitleSegment(
                start=segments[-1].start,
                end=max(segments[-1].end, end),
                text=text,
            )
            continue
        segments.append(SubtitleSegment(start=start, end=max(start, end), text=text))
    return segments


def load_subtitles_from_file(
    path: Path,
    unmask_finance: bool = False,
) -> list[SubtitleSegment]:
    """根据文件扩展名加载字幕列表。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt_text(raw, unmask_finance=unmask_finance)
    if suffix == ".vtt":
        return parse_vtt_text(raw, unmask_finance=unmask_finance)
    if suffix == ".json":
        payload = json.loads(raw)
        items = payload.get("segments", payload) if isinstance(payload, dict) else payload
        segments: list[SubtitleSegment] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if unmask_finance:
                text = restore_masked_terms(text)
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            segments.append(SubtitleSegment(start=start, end=max(start, end), text=text))
        return segments
    raise ValueError(f"Unsupported subtitle file format: {path}")


def estimate_tokens(text: str) -> int:
    """估算中英文混合文本的 Token 数（中文字符约 1.5 tokens，英文词/符号约 0.35 tokens/char）。"""
    if not text:
        return 0
    cjk_count = len(_CJK_CHAR_RE.findall(text))
    other_count = len(text) - cjk_count
    return max(1, int(cjk_count * 1.5 + other_count * 0.35))


def slice_segments_by_range(
    segments: list[SubtitleSegment],
    start_sec: float,
    end_sec: float,
) -> list[SubtitleSegment]:
    """提取与 ``[start_sec, end_sec]`` 有时间交集的字幕片段。"""
    return [
        seg
        for seg in segments
        if seg.end >= start_sec and seg.start <= end_sec
    ]
