"""Layer 2 检索核心：面向中英文混合字幕的 BM25 检索与时间窗聚簇合并。

原理说明：
- 纯英文空格分词在中文场景下会失效；完全依赖重型向量模型则会增加冷启动和安装门槛。
- 这里采用“英文/数字整词 + 中文单字与双字（Bigram）”混合分词器配合 BM25，
  既支持精确术语（如 ``RMSNorm``、``LayerNorm``、``PyTorch``），也能高召回中文短语。
- 命中单句后向前后扩展上下文窗口，并把相邻重叠窗口合并，返回给 Agent 最紧凑的候选区间。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import MomentHit, SubtitleSegment, format_timestamp

_ALNUM_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_bilingual(text: str) -> list[str]:
    """将中英文混合文本切分为适合 BM25 匹配的词元列表。"""
    lowered = text.lower()
    tokens: list[str] = _ALNUM_RE.findall(lowered)
    for run in _CJK_RUN_RE.findall(lowered):
        chars = list(run)
        tokens.extend(chars)
        if len(chars) >= 2:
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return tokens


def _merge_candidate_windows(
    candidates: list[dict],
    merge_gap_sec: float = 5.0,
) -> list[dict]:
    """将时间重叠或间隔很近的命中窗口合并，避免返回冗余重复区间。"""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item["start"])
    merged: list[dict] = [ordered[0]]
    for item in ordered[1:]:
        prev = merged[-1]
        if item["start"] <= prev["end"] + merge_gap_sec:
            prev["end"] = max(prev["end"], item["end"])
            prev["score"] = max(prev["score"], item["score"])
            prev["matched_terms"] = sorted(set(prev["matched_terms"]) | set(item["matched_terms"]))
        else:
            merged.append(item)
    return merged


def search_moments(
    segments: list[SubtitleSegment],
    query: str,
    duration_sec: float = 0.0,
    top_k: int = 5,
    window_sec: float = 20.0,
    min_score: float = 0.05,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[MomentHit]:
    """在字幕片段上执行 BM25 检索，返回按相关度排序的时间窗证据。"""
    query_tokens = tokenize_bilingual(query)
    if not segments or not query_tokens:
        return []

    doc_tokens_list = [tokenize_bilingual(seg.text) for seg in segments]
    doc_freqs: Counter[str] = Counter()
    for tokens in doc_tokens_list:
        doc_freqs.update(set(tokens))

    total_docs = len(segments)
    avg_len = sum(len(t) for t in doc_tokens_list) / max(1, total_docs)
    unique_query_terms = list(dict.fromkeys(query_tokens))

    scored: list[tuple[int, float, list[str]]] = []
    for idx, doc_tokens in enumerate(doc_tokens_list):
        if not doc_tokens:
            continue
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        matched: list[str] = []

        for term in unique_query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            matched.append(term)
            df = doc_freqs[term]
            # 标准 BM25 IDF（加 1 平滑，避免小语料下出现负值）
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1.0 - b + b * (doc_len / max(1e-6, avg_len)))
            score += idf * ((freq * (k1 + 1.0)) / max(1e-6, denom))

        # 精确子串命中额外加权，优先把原词完整出现的片段排在前面
        if query.strip().lower() in segments[idx].text.lower():
            score *= 1.35

        if score >= min_score and matched:
            scored.append((idx, score, matched))

    if not scored:
        return []

    # 先按分数取前 top_k * 3 个锚点扩展成时间窗，再合并相邻区间
    scored.sort(key=lambda item: item[1], reverse=True)
    half_win = max(2.0, window_sec / 2.0)
    raw_windows: list[dict] = []
    for idx, score, matched in scored[: max(top_k * 3, top_k)]:
        seg = segments[idx]
        mid = (seg.start + seg.end) / 2.0
        start = max(0.0, mid - half_win)
        end = mid + half_win
        if duration_sec > 0:
            end = min(duration_sec, end)
        raw_windows.append(
            {
                "start": round(start, 2),
                "end": round(max(start, end), 2),
                "score": score,
                "matched_terms": matched,
            }
        )

    merged = _merge_candidate_windows(raw_windows, merge_gap_sec=4.0)
    merged.sort(key=lambda item: item["score"], reverse=True)

    hits: list[MomentHit] = []
    for rank, win in enumerate(merged[:top_k], start=1):
        snippet_lines = [
            f"[{seg.start_str}] {seg.text}"
            for seg in segments
            if seg.end >= win["start"] and seg.start <= win["end"]
        ]
        snippet = " / ".join(snippet_lines)
        if len(snippet) > 360:
            snippet = snippet[:357] + "..."
        raw_terms = win["matched_terms"]
        preferred_terms = [t for t in raw_terms if len(t) > 1 or t.isascii()]
        display_terms = (preferred_terms or raw_terms)[:8]
        hits.append(
            MomentHit(
                rank=rank,
                start_sec=win["start"],
                end_sec=win["end"],
                start_ts=format_timestamp(win["start"]),
                end_ts=format_timestamp(win["end"]),
                score=round(win["score"], 4),
                matched_terms=display_terms,
                snippet=snippet,
            )
        )
    return hits
