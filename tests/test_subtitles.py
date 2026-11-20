"""字幕解析、词表边界还原与时间切片测试。"""

from __future__ import annotations

import json
from pathlib import Path

from vidctx.subtitles import (
    estimate_tokens,
    load_subtitles_from_file,
    parse_srt_text,
    parse_vtt_text,
    restore_masked_terms,
    slice_segments_by_range,
)


def test_parse_srt_with_bom_and_html_tags() -> None:
    raw = (
        "\ufeff1\n"
        "00:00:01,500 --> 00:00:04,200\n"
        "<b>欢迎来到</b> RMSNorm 讲解\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "第二行内容\n"
    )
    segments = parse_srt_text(raw)
    assert len(segments) == 2
    assert segments[0].start == 1.5
    assert segments[0].end == 4.2
    assert segments[0].text == "欢迎来到 RMSNorm 讲解"
    assert segments[1].text == "第二行内容"


def test_parse_vtt_deduplicates_rolling_captions() -> None:
    raw = (
        "WEBVTT\nKind: captions\nLanguage: zh-Hans\n\n"
        "00:00:10.000 --> 00:00:12.000 align:start\n"
        "今天分析A股和CEO发言\n\n"
        "00:00:12.100 --> 00:00:14.500\n"
        "今天分析A股和CEO发言\n\n"
        "00:00:15.000 --> 00:00:18.000\n"
        "很多人满C之后爆C了，L润全没了\n"
    )
    segments = parse_vtt_text(raw, unmask_finance=True)
    assert len(segments) == 2
    assert segments[0].start == 10.0
    assert segments[0].end == 14.5
    assert segments[0].text == "今天分析A股和CEO发言"
    assert segments[1].text == "很多人满仓之后爆仓了，利润全没了"


def test_restore_masked_terms_preserves_normal_acronyms() -> None:
    text = "A股的CEO站在C位谈G票N市与C底"
    restored = restore_masked_terms(text)
    assert "A股" in restored
    assert "CEO" in restored
    assert "C位" in restored
    assert "股票" in restored
    assert "牛市" in restored
    assert "抄底" in restored


def test_load_json_and_slice_range(tmp_path: Path) -> None:
    sub_file = tmp_path / "demo.json"
    sub_file.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "开场"},
                    {"start": 10.0, "end": 15.0, "text": "核心公式"},
                    {"start": 25.0, "end": 30.0, "text": "结尾"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segments = load_subtitles_from_file(sub_file)
    sliced = slice_segments_by_range(segments, 12.0, 20.0)
    assert len(sliced) == 1
    assert sliced[0].text == "核心公式"
    assert estimate_tokens("测试RMSNorm公式") > 0
