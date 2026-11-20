"""纯 Pillow 视觉信息密度评分与 dHash 感知去重测试。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from vidctx.vision import (
    compute_dhash,
    compute_info_density,
    hamming_distance,
    select_informative_frames,
)


def _make_blank_frame(path: Path) -> Path:
    img = Image.new("RGB", (320, 180), color=(15, 15, 15))
    img.save(path)
    return path


def _make_slide_frame(path: Path, variant: int = 1) -> Path:
    img = Image.new("RGB", (640, 360), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    if variant == 1:
        draw.rectangle([30, 30, 610, 80], fill=(30, 64, 175))
        for y in range(110, 320, 35):
            draw.line([50, y, 560, y], fill=(15, 23, 42), width=3)
            draw.rectangle([50, y - 10, 90, y + 10], fill=(220, 38, 38))
    else:
        draw.ellipse([80, 60, 280, 260], outline=(5, 150, 105), width=6)
        draw.polygon([(360, 280), (480, 70), (590, 280)], fill=(217, 119, 6))
        for x in range(320, 600, 40):
            draw.line([x, 40, x, 330], fill=(71, 85, 105), width=2)
    img.save(path)
    return path


def test_info_density_distinguishes_blank_and_rich_slides(tmp_path: Path) -> None:
    blank = _make_blank_frame(tmp_path / "blank.png")
    slide = _make_slide_frame(tmp_path / "slide.png", variant=1)

    blank_score = compute_info_density(blank)
    slide_score = compute_info_density(slide)
    assert blank_score < 0.1
    assert slide_score > 0.35
    assert slide_score > blank_score


def test_dhash_deduplicates_near_identical_frames(tmp_path: Path) -> None:
    slide1_a = _make_slide_frame(tmp_path / "s1_a.png", variant=1)
    slide1_b = _make_slide_frame(tmp_path / "s1_b.png", variant=1)
    # 轻微扰动（加一个极小像素点）仍应被判定为近似重复帧
    with Image.open(slide1_b) as im:
        im.putpixel((10, 10), (200, 200, 200))
        im.save(slide1_b)
    slide2 = _make_slide_frame(tmp_path / "s2.png", variant=2)

    assert hamming_distance(compute_dhash(slide1_a), compute_dhash(slide1_b)) < 6
    assert hamming_distance(compute_dhash(slide1_a), compute_dhash(slide2)) >= 6

    selected = select_informative_frames(
        [(1.0, slide1_a), (2.0, slide1_b), (5.0, slide2)],
        max_frames=4,
    )
    assert len(selected) == 2
    assert selected[0].timestamp_sec < selected[1].timestamp_sec
