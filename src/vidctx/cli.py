"""CLI 入口：提供 ``vidctx peek / search / inspect`` 三个渐进式子命令。"""

from __future__ import annotations

import json
import sys

import click

from .engine import VideoContextEngine


def _emit_json(payload: object) -> None:
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Progressive Video Context Reader for AI Agents (CLI)."""


@cli.command("peek")
@click.argument("source")
@click.option("--max-chapters", default=12, show_default=True, type=int, help="最大章节切片数量。")
@click.option("--unmask-finance", is_flag=True, help="启用常见财经/行业字母遮罩词还原。")
@click.option("--refresh", is_flag=True, help="忽略本地索引缓存重新解析。")
@click.option(
    "--fallback-index",
    type=click.Choice(["none", "ocr", "asr", "auto"], case_sensitive=False),
    default="none",
    show_default=True,
    help="当视频无文本字幕时，按需通过底部 OCR 或语音识别构建本地缓存索引。",
)
@click.option("--json", "as_json", is_flag=True, help="以结构化 JSON 输出。")
def peek_cmd(
    source: str,
    max_chapters: int,
    unmask_finance: bool,
    refresh: bool,
    fallback_index: str,
    as_json: bool,
) -> None:
    """Layer 1：低成本返回视频章节地图与 Token 估算。"""
    engine = VideoContextEngine()
    try:
        result = engine.peek_video_map(
            source=source,
            max_chapters=max_chapters,
            unmask_finance=unmask_finance,
            refresh=refresh,
            fallback_index=fallback_index.lower(),  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        _emit_json(result.model_dump())
        return

    click.echo(f"[{result.video_id}] {result.title} ({result.duration_ts})")
    click.echo(
        f"Source: {result.source_type} | Subtitles: {result.subtitle_source} "
        f"| Tokens(Full vs Map): {result.estimated_full_tokens} -> {result.map_tokens}"
    )
    click.echo("")
    for ch in result.chapters:
        click.echo(f"  {ch.index:02d}. [{ch.start_ts} - {ch.end_ts}] {ch.title}: {ch.preview}")
    click.echo(f"\nHint: {result.next_step_hint}")


@cli.command("search")
@click.argument("source")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True, type=int, help="返回最相关的前 K 个时间窗。")
@click.option("--window-sec", default=20.0, show_default=True, type=float, help="单次命中扩展窗口宽度（秒）。")
@click.option("--unmask-finance", is_flag=True, help="启用常见财经/行业字母遮罩词还原。")
@click.option(
    "--fallback-index",
    type=click.Choice(["none", "ocr", "asr", "auto"], case_sensitive=False),
    default="none",
    show_default=True,
    help="当视频无文本字幕时，按需通过底部 OCR 或语音识别构建本地缓存索引。",
)
@click.option("--json", "as_json", is_flag=True, help="以结构化 JSON 输出。")
def search_cmd(
    source: str,
    query: str,
    top_k: int,
    window_sec: float,
    unmask_finance: bool,
    fallback_index: str,
    as_json: bool,
) -> None:
    """Layer 2：按关键词/语义检索相关视频时间段。"""
    engine = VideoContextEngine()
    try:
        result = engine.search_video_moments(
            source=source,
            query=query,
            top_k=top_k,
            window_sec=window_sec,
            unmask_finance=unmask_finance,
            fallback_index=fallback_index.lower(),  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        _emit_json(result.model_dump())
        return

    click.echo(f"Query: {result.query!r} | Segments searched: {result.total_segments_searched}")
    if not result.hits:
        click.echo(result.note or "No matching moments found.")
        return
    for hit in result.hits:
        click.echo(
            f"  #{hit.rank} [{hit.start_ts} - {hit.end_ts}] "
            f"({hit.start_sec:.1f}s - {hit.end_sec:.1f}s) score={hit.score}"
        )
        click.echo(f"     {hit.snippet}")


@cli.command("inspect")
@click.argument("source")
@click.option("--start", "start_sec", required=True, type=float, help="起始时间（秒）。")
@click.option("--end", "end_sec", required=True, type=float, help="结束时间（秒）。")
@click.option("--max-frames", default=4, show_default=True, type=int, help="最多保留的关键帧数量。")
@click.option("--unmask-finance", is_flag=True, help="启用常见财经/行业字母遮罩词还原。")
@click.option("--json", "as_json", is_flag=True, help="以结构化 JSON 输出。")
def inspect_cmd(
    source: str,
    start_sec: float,
    end_sec: float,
    max_frames: int,
    unmask_finance: bool,
    as_json: bool,
) -> None:
    """Layer 3：针对窄时间窗精确抽帧、评分去重并提取原话。"""
    engine = VideoContextEngine()
    try:
        result = engine.inspect_frame_range(
            source=source,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
            unmask_finance=unmask_finance,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        _emit_json(result.model_dump())
        return

    click.echo(f"[{result.video_id}] Window: {result.window_ts}")
    for w in result.warnings:
        click.echo(f"  WARN: {w}", file=sys.stderr)
    click.echo("\nTranscript:")
    for seg in result.transcript_lines:
        click.echo(f"  [{seg.start_str}] {seg.text}")
    click.echo("\nSelected Keyframes:")
    for frame in result.frames:
        click.echo(
            f"  [{frame.timestamp_ts}] score={frame.info_score} dhash={frame.dhash} "
            f"path={frame.image_path}"
        )
        if frame.nearest_subtitle:
            click.echo(f"     caption: {frame.nearest_subtitle}")


def main() -> None:
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
