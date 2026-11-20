"""FastMCP 服务端入口：向 AI Agent 暴露三层按需视频上下文工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image

from .engine import VideoContextEngine


def create_server(engine: VideoContextEngine | None = None) -> FastMCP:
    """创建并注册 ``vidctx-mcp`` 的三个核心渐进式工具。

    支持注入自定义 ``VideoContextEngine``（便于单元测试或切换后端）。
    """
    active_engine = engine or VideoContextEngine()
    mcp = FastMCP(
        name="vidctx-mcp",
        instructions=(
            "Progressive Video Context Reader for AI Agents. "
            "Always follow the 3-step context-saving workflow: "
            "1) Call `peek_video_map` first to inspect duration, chapters, and token budget; "
            "2) Call `search_video_moments` to locate candidate time ranges via BM25; "
            "3) Call `inspect_frame_range` ONLY on targeted 10-20s intervals to inspect dialogue and keyframes."
        ),
    )

    @mcp.tool()
    def peek_video_map(
        source: str,
        max_chapters: int = 12,
        max_preview_chars: int = 72,
        unmask_finance: bool = False,
        refresh: bool = False,
        fallback_index: Literal["none", "ocr", "asr", "auto"] = "none",
    ) -> dict:
        """Layer 1: 低 Token 返回视频目录结构、时长与字幕概览（不解码画面）。"""
        result = active_engine.peek_video_map(
            source=source,
            max_chapters=max_chapters,
            max_preview_chars=max_preview_chars,
            unmask_finance=unmask_finance,
            refresh=refresh,
            fallback_index=fallback_index,
        )
        return result.model_dump()

    @mcp.tool()
    def search_video_moments(
        source: str,
        query: str,
        top_k: int = 5,
        window_sec: float = 20.0,
        unmask_finance: bool = False,
        fallback_index: Literal["none", "ocr", "asr", "auto"] = "none",
    ) -> dict:
        """Layer 2: 在视频字幕索引中检索与查询词最匹配的时间区间。"""
        result = active_engine.search_video_moments(
            source=source,
            query=query,
            top_k=top_k,
            window_sec=window_sec,
            unmask_finance=unmask_finance,
            fallback_index=fallback_index,
        )
        return result.model_dump()

    @mcp.tool(structured_output=False)
    def inspect_frame_range(
        source: str,
        start_sec: float,
        end_sec: float,
        max_frames: int = 4,
        include_images: bool = True,
        unmask_finance: bool = False,
    ) -> list[str | Image]:
        """Layer 3: 仅针对指定时间区间精确抽帧、评分去重并提取带时间戳原话。"""
        bundle = active_engine.inspect_frame_range(
            source=source,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
            unmask_finance=unmask_finance,
        )
        items: list[str | Image] = [json.dumps(bundle.model_dump(), ensure_ascii=False)]
        if include_images:
            for frame in bundle.frames:
                path = Path(frame.image_path)
                if path.is_file():
                    items.append(Image(path=path))
        return items

    return mcp


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
