"""受控本地缓存与临时工作区管理（带容量上限与自动淘汰）。

解决传统脚本在长驻 MCP Server 模式下反复 ``mkdtemp`` 导致磁盘膨胀的问题：
- 每个 ``video_id`` 拥有独立缓存目录；
- 每次命中或写入都会更新访问时间戳；
- 当缓存总大小超过 ``max_bytes`` 时，按 LRU（最近最少使用）策略自动清理最旧的视频目录。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .models import MediaInfo, SubtitleSegment


class WorkspaceCache:
    """基于文件系统的视频元信息、字幕与局部帧缓存管理器。"""

    def __init__(
        self,
        root_dir: Path | None = None,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        default_root = Path(os.environ.get("VIDCTX_CACHE_DIR", Path.home() / ".cache" / "vidctx"))
        self.root_dir = (root_dir or default_root).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(10 * 1024 * 1024, max_bytes)

    def video_dir(self, video_id: str) -> Path:
        path = self.root_dir / video_id
        path.mkdir(parents=True, exist_ok=True)
        self._touch(path)
        return path

    def frames_dir(self, video_id: str, start_sec: float, end_sec: float) -> Path:
        tag = f"frames_{int(start_sec * 10)}_{int(end_sec * 10)}"
        path = self.video_dir(video_id) / tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_index(self, video_id: str) -> tuple[MediaInfo, list[SubtitleSegment]] | None:
        index_file = self.root_dir / video_id / "index.json"
        if not index_file.is_file():
            return None
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
            info = MediaInfo.model_validate(payload["info"])
            segments = [SubtitleSegment.model_validate(s) for s in payload["segments"]]
            self._touch(index_file.parent)
            return info, segments
        except Exception:
            return None

    def save_index(
        self,
        video_id: str,
        info: MediaInfo,
        segments: list[SubtitleSegment],
    ) -> None:
        vdir = self.video_dir(video_id)
        index_file = vdir / "index.json"
        payload = {
            "info": info.model_dump(),
            "segments": [s.model_dump() for s in segments],
        }
        index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.evict_if_needed(preserve=video_id)

    def evict_if_needed(self, preserve: str | None = None) -> list[str]:
        """当缓存总体积超出阈值时，按 LRU 顺序清理旧目录并返还已删除的 ``video_id``。"""
        entries: list[tuple[float, int, Path]] = []
        total_size = 0
        for child in self.root_dir.iterdir():
            if not child.is_dir():
                continue
            size = self._dir_size(child)
            total_size += size
            mtime = self._last_touched(child)
            entries.append((mtime, size, child))

        if total_size <= self.max_bytes:
            return []

        entries.sort(key=lambda item: item[0])
        removed: list[str] = []
        for _, size, child in entries:
            if total_size <= self.max_bytes:
                break
            if preserve and child.name == preserve:
                continue
            shutil.rmtree(child, ignore_errors=True)
            total_size -= size
            removed.append(child.name)
        return removed

    @staticmethod
    def _touch(path: Path) -> None:
        marker = path / ".last_used"
        try:
            marker.touch()
        except OSError:
            pass

    @staticmethod
    def _last_touched(path: Path) -> float:
        marker = path / ".last_used"
        target = marker if marker.exists() else path
        try:
            return target.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total
