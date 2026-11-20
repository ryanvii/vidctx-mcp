"""双语 BM25 检索与相邻窗口合并测试。"""

from __future__ import annotations

from vidctx.models import SubtitleSegment
from vidctx.search import search_moments, tokenize_bilingual


def test_tokenize_bilingual_splits_cjk_and_english() -> None:
    tokens = tokenize_bilingual("深入理解 RMSNorm 归一化")
    assert "rmsnorm" in tokens
    assert "归" in tokens
    assert "归一" in tokens
    assert "一化" in tokens


def test_search_moments_ranks_relevant_window_first() -> None:
    segments = [
        SubtitleSegment(start=0.0, end=8.0, text="大家好，今天先介绍一下课程安排和开发环境。"),
        SubtitleSegment(start=10.0, end=18.0, text="我们用 Python 搭建测试脚本。"),
        SubtitleSegment(start=120.0, end=128.0, text="接下来看 RMSNorm 的数学推导：它省去了减均值步骤。"),
        SubtitleSegment(start=129.0, end=136.0, text="只做均方根缩放，计算速度明显提升。"),
        SubtitleSegment(start=300.0, end=310.0, text="最后总结一下今天的作业。"),
    ]
    hits = search_moments(segments, query="RMSNorm 数学推导", duration_sec=320.0, top_k=3, window_sec=16.0)
    assert len(hits) >= 1
    top = hits[0]
    assert top.start_sec <= 125.0 <= top.end_sec
    assert "RMSNorm" in top.snippet
    # 相邻的 120s 和 129s 应该被合并为一个时间窗，而不是返回两条重复窗口
    overlapping = [h for h in hits if 110.0 <= h.start_sec <= 140.0]
    assert len(overlapping) == 1


def test_search_moments_returns_empty_when_no_match() -> None:
    segments = [SubtitleSegment(start=0.0, end=5.0, text="今天天气不错")]
    assert search_moments(segments, query="Kubernetes", duration_sec=10.0) == []


def test_search_moments_keeps_single_char_matched_terms_non_empty() -> None:
    segments = [SubtitleSegment(start=0.0, end=6.0, text="注意看这个权重的分配变化")]
    hits = search_moments(segments, query="除权", duration_sec=10.0)
    assert len(hits) == 1
    assert hits[0].matched_terms == ["权"]
