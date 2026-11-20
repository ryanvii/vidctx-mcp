"""CLI 子命令与 FastMCP 服务端协议端到端测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vidctx.cache import WorkspaceCache
from vidctx.cli import cli
from vidctx.engine import VideoContextEngine
from vidctx.server import create_server

from test_engine import FakeMediaBackend


def test_cli_peek_search_inspect_with_local_srt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDCTX_CACHE_DIR", str(tmp_path / "cli_cache"))
    srt_file = tmp_path / "talk.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\n今天分享订单簿失衡策略\n\n"
        "2\n00:00:10,000 --> 00:00:16,000\n当买盘深度远大于卖盘时观察短期价格冲击\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    res_peek = runner.invoke(cli, ["peek", str(srt_file), "--json"])
    assert res_peek.exit_code == 0, res_peek.output
    peek_data = json.loads(res_peek.output)
    assert peek_data["has_subtitles"] is True

    res_search = runner.invoke(cli, ["search", str(srt_file), "订单簿失衡", "--json"])
    assert res_search.exit_code == 0, res_search.output
    search_data = json.loads(res_search.output)
    assert len(search_data["hits"]) >= 1

    res_inspect = runner.invoke(
        cli,
        ["inspect", str(srt_file), "--start", "0", "--end", "12", "--json"],
    )
    assert res_inspect.exit_code == 0, res_inspect.output
    inspect_data = json.loads(res_inspect.output)
    assert len(inspect_data["transcript_lines"]) == 2
    assert any("Subtitle-only" in w for w in inspect_data["warnings"])


@pytest.mark.asyncio
async def test_fastmcp_server_tools_and_image_payload(tmp_path: Path) -> None:
    video_file = tmp_path / "demo.mp4"
    video_file.write_bytes(b"fake-video")

    engine = VideoContextEngine(
        cache=WorkspaceCache(root_dir=tmp_path / "mcp_cache"),
        backend=FakeMediaBackend(),
    )
    mcp = create_server(engine=engine)

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {"peek_video_map", "search_video_moments", "inspect_frame_range"}

    peek_blocks = await mcp.call_tool("peek_video_map", {"source": str(video_file)})
    map_meta = json.loads(peek_blocks[0].text)
    assert map_meta["title"] == "MiniMind 第7集：RMSNorm 原理"

    search_blocks = await mcp.call_tool(
        "search_video_moments",
        {"source": str(video_file), "query": "RMSNorm"},
    )
    search_meta = json.loads(search_blocks[0].text)
    assert len(search_meta["hits"]) >= 1

    content_blocks = await mcp.call_tool(
        "inspect_frame_range",
        {
            "source": str(video_file),
            "start_sec": 608.0,
            "end_sec": 625.0,
            "max_frames": 3,
            "include_images": True,
        },
    )
    types = [block.type for block in content_blocks]
    assert "text" in types
    assert "image" in types
    payload = json.loads(content_blocks[0].text)
    assert len(payload["frames"]) >= 1
