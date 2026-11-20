"""Layer 3 视觉关键帧评分与感知去重模块（纯 Pillow 实现，零重型 CV 依赖）。

原理说明：
1. 信息密度评分（``compute_info_density``）：
   结合边缘锐度（``ImageFilter.FIND_EDGES`` 标准差）、全局对比度（灰度标准差）和
   信息熵（64-bin 灰度直方图香农熵），过滤掉纯黑转场帧、模糊过渡帧和纯色背景帧。
2. 感知哈希去重（``compute_dhash`` + ``hamming_distance``）：
   采用 9x8 缩小灰度图计算相邻像素水平差分生成 64-bit dHash。同一页 PPT 连续讲 10 秒时，
   多帧之间的 dHash 汉明距离通常 <= 5，从而只保留最具代表性的一帧。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


@dataclass
class ScoredFrame:
    """带评分与感知哈希的候选帧。"""

    path: Path
    timestamp_sec: float
    info_score: float = 0.0
    dhash: str = ""


def compute_info_density(image_path: Path) -> float:
    """计算单张图片的信息密度分数，返回 ``[0.0, 1.0]``。"""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        # 缩放到统一宽度加速计算，同时避免分辨率差异干扰边缘统计
        if gray.width > 480:
            ratio = 480.0 / gray.width
            gray = gray.resize((480, max(1, int(gray.height * ratio))), Image.Resampling.BILINEAR)

        contrast_std = ImageStat.Stat(gray).stddev[0]
        contrast_score = min(1.0, contrast_std / 75.0)

        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_std = ImageStat.Stat(edges).stddev[0]
        edge_score = min(1.0, edge_std / 55.0)

        hist = gray.histogram()
        total_pixels = sum(hist) or 1
        # 将 256 级灰度合并为 64 个桶，降低微弱噪点对熵的虚高影响
        entropy = 0.0
        for bucket_start in range(0, 256, 4):
            bucket_count = sum(hist[bucket_start : bucket_start + 4])
            if bucket_count > 0:
                p = bucket_count / total_pixels
                entropy -= p * math.log2(p)
        entropy_score = min(1.0, entropy / 6.0)

        score = edge_score * 0.40 + contrast_score * 0.30 + entropy_score * 0.30
        return round(max(0.0, min(1.0, score)), 4)


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """计算 64-bit 差分感知哈希（dHash），返回 16 位十六进制字符串。"""
    with Image.open(image_path) as img:
        resized = img.convert("L").resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS,
        )
        raw_getter = getattr(resized, "get_flattened_data", None) or resized.getdata
        pixels = list(raw_getter())
        bits = 0
        width = hash_size + 1
        for row in range(hash_size):
            offset = row * width
            for col in range(hash_size):
                bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
        hex_width = (hash_size * hash_size) // 4
        return f"{bits:0{hex_width}x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """计算两个十六进制感知哈希的汉明距离。"""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def select_informative_frames(
    candidates: list[tuple[float, Path]],
    max_frames: int = 4,
    hash_threshold: int = 6,
) -> list[ScoredFrame]:
    """对区间内候选帧依次执行：信息密度打分 → 相似帧去重 → 按时间顺序返回 Top-N。"""
    if not candidates or max_frames <= 0:
        return []

    scored: list[ScoredFrame] = []
    for ts, path in candidates:
        if not path.exists():
            continue
        try:
            score = compute_info_density(path)
            dhash = compute_dhash(path)
        except Exception:
            continue
        scored.append(ScoredFrame(path=path, timestamp_sec=ts, info_score=score, dhash=dhash))

    if not scored:
        return []

    # 先按时间顺序去重，保留同一静态场景中信息密度最高的一帧
    scored.sort(key=lambda item: item.timestamp_sec)
    deduped: list[ScoredFrame] = []
    for frame in scored:
        if not deduped:
            deduped.append(frame)
            continue
        last = deduped[-1]
        if hamming_distance(frame.dhash, last.dhash) < hash_threshold:
            if frame.info_score > last.info_score:
                deduped[-1] = frame
        else:
            deduped.append(frame)

    # 若去重后仍多于 max_frames，按信息密度挑 Top-N 再恢复时间顺序
    if len(deduped) > max_frames:
        deduped.sort(key=lambda item: item.info_score, reverse=True)
        deduped = deduped[:max_frames]
        deduped.sort(key=lambda item: item.timestamp_sec)

    return deduped
