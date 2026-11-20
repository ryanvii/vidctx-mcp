"""无字幕视频的可选索引回退模块（局部 OCR 采样 & 语音识别 ASR）。

设计理念：
- 默认保持 Layer 1 (`peek_video_map`) 零下载、零算力消耗；
- 仅当发现视频完全无文本字幕、且调用方显式请求 ``fallback_index="ocr"|"asr"|"auto"`` 时，
  才执行一次性的离线建索引，并将结果写入 ``WorkspaceCache``，后续检索全部秒级复用；
- OCR 采用单次 ``ffmpeg`` 流式抽帧 + 底部字幕区裁剪，20 分钟视频通常仅需数秒即可完成采样。
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from .models import SubtitleSegment
from .subtitles import restore_masked_terms

_CJK_OR_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_OCR_SINGLETON = None


class FallbackIndexError(RuntimeError):
    """回退建索引（OCR / ASR）缺少依赖或执行失败时抛出。"""


def ocr_backend_available() -> bool:
    """检查当前环境是否安装了 RapidOCR。"""
    for mod in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def asr_backend_available() -> bool:
    """检查当前环境是否配置了 ASR API Key 或安装了本地 faster-whisper。"""
    if os.environ.get("VIDCTX_ASR_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        __import__("faster_whisper")
        return True
    except ImportError:
        return False


def _get_ocr_engine():
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    # 抑制 RapidOCR 日志向 stdout 输出，避免污染 MCP stdio JSON-RPC 或 CLI --json 管道
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from rapidocr import RapidOCR
            _OCR_SINGLETON = RapidOCR(params={"Global.log_level": "error"})
            return _OCR_SINGLETON
        except TypeError:
            from rapidocr import RapidOCR
            _OCR_SINGLETON = RapidOCR()
            return _OCR_SINGLETON
        except ImportError:
            pass
        try:
            from rapidocr_onnxruntime import RapidOCR as LegacyRapidOCR
            _OCR_SINGLETON = LegacyRapidOCR()
            return _OCR_SINGLETON
        except ImportError as exc:
            raise FallbackIndexError(
                "OCR fallback requires `rapidocr` + `onnxruntime`. "
                "Install with: pip install 'vidctx-mcp[ocr]'"
            ) from exc


def _ocr_single_image(engine, image_path: Path) -> str:
    with contextlib.redirect_stdout(sys.stderr):
        result = engine(str(image_path))
    # RapidOCR 3.x 返回带有 .txts 属性的对象；2.x 返回 (boxes_list, elapse)
    if hasattr(result, "txts"):
        txts = result.txts or []
        lines = [str(t).strip() for t in txts if str(t).strip()]
        return " ".join(lines)
    items = result[0] if isinstance(result, tuple) else result
    if not items:
        return ""
    lines: list[str] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            txt = str(item[1]).strip()
            if txt:
                lines.append(txt)
    return " ".join(lines)


def merge_ocr_samples(
    samples: list[tuple[float, str]],
    interval_sec: float,
    duration_sec: float = 0.0,
    similarity_threshold: float = 0.80,
) -> list[SubtitleSegment]:
    """将连续采样的字幕带 OCR 文本按相似度合并为带时间戳片段。"""
    segments: list[SubtitleSegment] = []
    current = ""
    start = 0.0
    end = 0.0
    miss_streak = 0

    def _flush() -> None:
        nonlocal current, miss_streak
        if not current:
            return
        stop = end + interval_sec
        if duration_sec > 0:
            stop = min(stop, duration_sec)
        segments.append(SubtitleSegment(start=round(start, 2), end=round(max(start, stop), 2), text=current))
        current = ""
        miss_streak = 0

    for ts, raw_text in samples:
        cleaned = " ".join((raw_text or "").split()).strip()
        if not _CJK_OR_WORD_RE.search(cleaned):
            cleaned = ""

        if not cleaned:
            if current:
                miss_streak += 1
                if miss_streak > 1:
                    _flush()
            continue

        miss_streak = 0
        if not current:
            current, start, end = cleaned, ts, ts
            continue

        ratio = SequenceMatcher(None, current.lower(), cleaned.lower()).ratio()
        if ratio >= similarity_threshold or cleaned in current or current in cleaned:
            if len(cleaned) > len(current):
                current = cleaned
            end = ts
        else:
            _flush()
            current, start, end = cleaned, ts, ts

    _flush()
    return segments


def index_via_ocr(
    media_path: Path,
    work_dir: Path,
    duration_sec: float = 0.0,
    ffmpeg_bin: str = "ffmpeg",
    interval_sec: float | None = None,
    bottom_ratio: float = 0.24,
    unmask_finance: bool = False,
) -> list[SubtitleSegment]:
    """通过单次 ffmpeg 底部裁剪抽帧 + RapidOCR 为无字幕视频构建可检索字幕索引。"""
    if not shutil.which(ffmpeg_bin):
        raise FallbackIndexError(f"Missing required binary '{ffmpeg_bin}' for OCR indexing.")

    engine = _get_ocr_engine()
    if interval_sec is None:
        # 20 分钟视频约采样 ~240 张字幕带小图，兼顾速度与召回
        interval_sec = max(2.5, min(6.0, (duration_sec / 220.0) if duration_sec > 0 else 4.0))

    ocr_dir = work_dir / "ocr_samples"
    if ocr_dir.exists():
        shutil.rmtree(ocr_dir, ignore_errors=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    ratio = max(0.12, min(0.5, bottom_ratio))
    vf = f"fps=1/{interval_sec:.3f},crop=iw:ih*{ratio:.2f}:0:ih*{1.0 - ratio:.2f},scale=960:-2"
    cmd = [
        ffmpeg_bin,
        "-i", str(media_path),
        "-vf", vf,
        "-q:v", "3",
        str(ocr_dir / "cap_%05d.jpg"),
        "-y", "-loglevel", "error",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise FallbackIndexError(f"ffmpeg caption-band sampling failed: {proc.stderr.strip()[:300]}")

    frame_files = sorted(ocr_dir.glob("cap_*.jpg"))
    if not frame_files:
        return []

    samples: list[tuple[float, str]] = []
    for idx, fpath in enumerate(frame_files):
        ts = round(idx * interval_sec, 2)
        text = _ocr_single_image(engine, fpath)
        samples.append((ts, text))

    # OCR 完成后清理临时采样图，保持缓存紧凑
    shutil.rmtree(ocr_dir, ignore_errors=True)

    segments = merge_ocr_samples(samples, interval_sec=interval_sec, duration_sec=duration_sec)
    if unmask_finance:
        segments = [
            SubtitleSegment(start=s.start, end=s.end, text=restore_masked_terms(s.text))
            for s in segments
        ]
    return segments


def index_via_asr(
    media_path: Path,
    work_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
    unmask_finance: bool = False,
) -> list[SubtitleSegment]:
    """提取紧凑单声道音频并通过 OpenAI 兼容 ASR API 或本地 faster-whisper 生成带时间戳字幕。"""
    if not shutil.which(ffmpeg_bin):
        raise FallbackIndexError(f"Missing required binary '{ffmpeg_bin}' for ASR indexing.")

    audio_path = work_dir / "asr_audio.mp3"
    cmd = [
        ffmpeg_bin,
        "-i", str(media_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        str(audio_path),
        "-y", "-loglevel", "error",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not audio_path.is_file():
        raise FallbackIndexError(f"ffmpeg audio extraction failed: {proc.stderr.strip()[:300]}")

    try:
        segments = _transcribe_audio_file(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    if unmask_finance:
        segments = [
            SubtitleSegment(start=s.start, end=s.end, text=restore_masked_terms(s.text))
            for s in segments
        ]
    return segments


def _transcribe_audio_file(audio_path: Path) -> list[SubtitleSegment]:
    api_key = os.environ.get("VIDCTX_ASR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        base_url = os.environ.get("VIDCTX_ASR_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
        model = os.environ.get("VIDCTX_ASR_MODEL", "whisper-1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        with open(audio_path, "rb") as fh:
            resp = client.audio.transcriptions.create(
                model=model,
                file=fh,
                response_format="verbose_json",
            )
        raw_segments = getattr(resp, "segments", None) or []
        segments: list[SubtitleSegment] = []
        for item in raw_segments:
            text = str(item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0.0) if isinstance(item, dict) else getattr(item, "start", 0.0))
            end = float(item.get("end", start) if isinstance(item, dict) else getattr(item, "end", start))
            segments.append(SubtitleSegment(start=round(start, 2), end=round(max(start, end), 2), text=text))
        if segments:
            return segments
        full_text = str(getattr(resp, "text", "") or "").strip()
        return [SubtitleSegment(start=0.0, end=0.0, text=full_text)] if full_text else []

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise FallbackIndexError(
            "ASR fallback requires either VIDCTX_ASR_API_KEY / OPENAI_API_KEY "
            "or local `faster-whisper` (`pip install 'vidctx-mcp[asr]'`)."
        ) from exc

    model_size = os.environ.get("VIDCTX_WHISPER_MODEL", "small")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(audio_path), vad_filter=True)
    return [
        SubtitleSegment(
            start=round(float(seg.start), 2),
            end=round(max(float(seg.start), float(seg.end)), 2),
            text=seg.text.strip(),
        )
        for seg in raw_segments
        if seg.text.strip()
    ]


def build_fallback_segments(
    mode: Literal["ocr", "asr", "auto"],
    media_path: Path,
    work_dir: Path,
    duration_sec: float = 0.0,
    ffmpeg_bin: str = "ffmpeg",
    unmask_finance: bool = False,
) -> tuple[list[SubtitleSegment], str]:
    """按指定模式构建回退字幕索引，返回 ``(segments, subtitle_source)``。"""
    if mode == "ocr":
        segs = index_via_ocr(
            media_path=media_path,
            work_dir=work_dir,
            duration_sec=duration_sec,
            ffmpeg_bin=ffmpeg_bin,
            unmask_finance=unmask_finance,
        )
        return segs, "fallback_ocr"

    if mode == "asr":
        segs = index_via_asr(
            media_path=media_path,
            work_dir=work_dir,
            ffmpeg_bin=ffmpeg_bin,
            unmask_finance=unmask_finance,
        )
        return segs, "fallback_asr"

    # auto：优先本地可用且低成本的途径
    errors: list[str] = []
    if asr_backend_available():
        try:
            segs = index_via_asr(
                media_path=media_path,
                work_dir=work_dir,
                ffmpeg_bin=ffmpeg_bin,
                unmask_finance=unmask_finance,
            )
            if segs:
                return segs, "fallback_asr"
        except Exception as exc:
            errors.append(f"asr: {exc}")

    if ocr_backend_available():
        try:
            segs = index_via_ocr(
                media_path=media_path,
                work_dir=work_dir,
                duration_sec=duration_sec,
                ffmpeg_bin=ffmpeg_bin,
                unmask_finance=unmask_finance,
            )
            if segs:
                return segs, "fallback_ocr"
        except Exception as exc:
            errors.append(f"ocr: {exc}")

    if errors:
        raise FallbackIndexError("; ".join(errors))
    raise FallbackIndexError(
        "No fallback indexing backend available. Install `vidctx-mcp[ocr]` or `vidctx-mcp[asr]`, "
        "or set VIDCTX_ASR_API_KEY."
    )
