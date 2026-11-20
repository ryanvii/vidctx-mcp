# vidctx-mcp — 面向 AI Agent 的按需渐进式视频上下文读取器

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **我解决的不是“如何把视频下载下来”，而是“如何让 AI Agent 在有限上下文预算内，高准确率地检索视频证据”。**

![vidctx-mcp 三层渐进式视频上下文演示](assets/demo.gif)

---

## 为什么需要渐进式视频上下文？

传统视频总结工具或普通 MCP 插件往往采用**暴力全量注入**：要么把 2 小时视频的 3 万字逐字稿一次性塞进上下文，要么均匀抽取几十张高清大图直接喂给多模态模型。这会带来三个严重问题：
1. **Token 爆炸与注意力稀释**：无效寒暄和冗余画面淹没了真正的关键帧和关键原话；
2. **带宽与算力浪费**：明明只想确认视频里第 14 分钟某个公式推导，却被迫解码甚至转录整部视频；
3. **证据不可追溯**：经过中间一层 LLM 二次总结后，原始时间戳和画面依据常常丢失或产生幻觉。

**`vidctx-mcp`** 将视频读取设计为符合 **Context Engineering（上下文工程）** 的三层按需漏斗：

```text
Layer 1: peek_video_map        ──► 仅返回目录/时间切片 + 全文 Token 预算估算（零像素解码）
             │
             ▼
Layer 2: search_video_moments  ──► 双语 BM25 检索（中文单双字 + 英文整词）+ 滑动窗口聚簇合并
             │
             ▼
Layer 3: inspect_frame_range   ──► 仅针对命中区间精确抽帧 + 纯 Pillow 信息密度打分 + dHash 感知去重 + 原话对齐
```

### 真实实测表现（22 分钟 B 站硬字幕量化教学视频）

- **Layer 1 目录映射**：全片 199 条 OCR 字幕预计 `11,677 tokens` → Layer 1 结构地图仅消耗 `1,792 tokens`（减少约 **85%** 初始上下文消耗）；
- **Layer 2 关键词秒级检索**：检索 `"除权 复权"` 精确命中 `00:18:23 - 00:19:19`（`score=14.0622`）；
- **Layer 3 窄窗抽帧取证**：仅对命中的 `00:18:30 - 00:18:45` 抽取并去重保留 2 张高信息密度关键帧，精确对齐原话与屏幕图解。

---

## 支持矩阵与维护边界（Supported Sources & Non-Goals）

| 输入类型 | 支持程度 | 说明 |
| :--- | :--- | :--- |
| **本地视频**（`.mp4/.mkv/.mov/.webm` 等） | ✅ 完整支持 | 自动读取同名旁挂或内嵌字幕，抽帧走 `ffmpeg` 快速定位 |
| **本地音频/播客**（`.mp3/.wav/.m4a/.flac/.ogg`） | ✅ 支持转录检索 | 搭配 `--fallback-index asr` 建索引；Layer 3 自动跳过抽帧并输出原话 |
| **纯字幕/转录**（`.srt/.vtt/.json`） | ✅ 零依赖支持 | 无需 `ffmpeg` 或 `yt-dlp`，直接构建 BM25 检索索引 |
| **YouTube / B站 / X 等公开链接** | ✅ 通过 `yt-dlp` | 自动剥离跟踪参数并限制 `--no-playlist`；YouTube 额外支持 `youtube-transcript-api` 回退 |
| **高风控 App 逆向爬虫（如无登录反爬绕过）** | ❌ 明确不做 | 不在仓库中维护易碎的私有网页签名/风控逆向代码，保障长期低维护可用性 |

---

## 核心特性与工程设计

- **严格维护边界与免责声明**：
  - 核心引擎仅面向**本地视频/音频/字幕文件**（`.mp4/.mkv/.srt/.vtt/.json`）及**可选稳定的公开上游链接**（严格委托给用户本地安装的 `yt-dlp`，且 Layer 1/2 默认 `--skip-download` 不下载视频本体）。
  - **不包含**任何高风控平台的 Cookie 逆向或网页反爬代码，确保仓库核心代码长期稳定、近零维护成本。
- **双语高精度 BM25 检索**（[`search.py`](src/vidctx/search.py)）：
  - 不依赖重型向量数据库或本地 Embedding 模型，使用“英文整词 + 中文单双字切分”混合分词 + BM25 + 精确子串加权 + 相邻窗口合并。
- **纯 Pillow 视觉质量筛选与感知去重**（[`vision.py`](src/vidctx/vision.py)）：
  - 无需强制安装 OpenCV/NumPy：通过边缘密度（`FIND_EDGES`）、全局对比度和灰度直方图香农熵综合打分，剔除纯黑/转场模糊帧；再用 64-bit dHash 感知哈希去除同一张 PPT 的重复截图。
- **受控 LRU 工作区缓存**（[`cache.py`](src/vidctx/cache.py)）：
  - 针对长驻 MCP Server 场景自动管理缓存容量（默认上限 512MB），超出阈值按最近最少使用（LRU）淘汰旧视频目录，避免磁盘泄漏。
- **无 `SIGALRM` 跨平台异步安全**：
  - 全部外部命令调用统一使用带显式超时的子进程接口，兼容异步 MCP 事件循环与 macOS/Linux/Windows。

---

## 安装与配置

### 1. 安装（推荐在独立虚拟环境中安装）

```bash
pip install -e .
# 如需支持远程公开视频链接（依赖 yt-dlp）：
pip install -e ".[remote]"
# 如需针对无字幕视频启用局部字幕带 OCR 或本地语音建索引：
pip install -e ".[ocr]"   # RapidOCR + ONNXRuntime
pip install -e ".[asr]"   # faster-whisper / OpenAI 兼容 ASR API
```

> 抽帧与探测本地视频元数据需要系统已安装 `ffmpeg` / `ffprobe`；若仅解析 `.srt` / `.vtt` / `.json` 字幕文件则完全零外部二进制依赖。

#### 可选环境变量（Cookie 透传与 ASR 服务）

- `VIDCTX_COOKIES_FROM_BROWSER=chrome` 或 `VIDCTX_COOKIES_FILE=/path/to/cookies.txt`：原生透传给 `yt-dlp`，用于在 B 站等需要登录态才能拉取 CC/AI 字幕的平台获取字幕。
- `VIDCTX_ASR_API_KEY` / `VIDCTX_ASR_BASE_URL` / `VIDCTX_ASR_MODEL`：可选 OpenAI 兼容语音转文字 API（当视频完全无字幕且使用 `--fallback-index asr` 时调用）。

### 2. CLI 使用示例

```bash
# Layer 1：查看视频结构地图与 Token 节省对比
vidctx peek ./lecture.mp4 --json

# 若视频无外挂/平台字幕（如部分 B 站硬字幕教程），可按需构建并缓存 OCR 索引
vidctx peek ./hardsub_video.mp4 --fallback-index ocr

# Layer 2：检索关键概念出现的时间窗口
vidctx search ./lecture.mp4 "RMSNorm 数学推导" --top-k 3

# Layer 3：针对命中窗口精确抽帧并对齐原话
vidctx inspect ./lecture.mp4 --start 810 --end 830 --max-frames 3
```

### 3. 在 Claude Code / Cursor / MCP Client 中接入

在你的 `mcp.json` 中添加：

```json
{
  "mcpServers": {
    "vidctx": {
      "command": "vidctx-mcp"
    }
  }
}
```

---

## 运行测试

```bash
pytest -v
```

测试套件通过注入确定性的测试媒体后端（生成真实 Pillow 测试图像与多格式字幕样本），可在无外网、无系统二进制依赖下完整验证三层检索、视觉评分去重、LRU 缓存淘汰及 FastMCP 协议调用链路。

---

## 免责与使用边界

本工具仅用于处理用户本地拥有合法权利的音视频/字幕文件，或通过用户自行安装的合规上游工具读取公开授权内容。请遵守各内容平台的服务条款与版权规范。

---

## 致谢与上游开源项目

本项目站在以下优秀开源项目的基础上构建，核心仅专注“面向 Agent 的三层渐进式上下文裁剪与视觉证据筛选”：

| 项目 | 作用 | 许可证 |
| :--- | :--- | :--- |
| [MCP Python SDK (`mcp`)](https://github.com/modelcontextprotocol/python-sdk) | FastMCP 服务端与标准协议通信 | MIT |
| [FFmpeg / FFprobe](https://ffmpeg.org/) | 本地视频元数据探测与窄时间窗快速抽帧 | LGPL / GPL |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | 公开平台视频元信息、字幕与按需媒体提取 | Unlicense |
| [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | YouTube 公开字幕快速回退获取 | MIT |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) & [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 无字幕视频可选的底部硬字幕 OCR 与语音转录索引 | Apache 2.0 / MIT |
| [Pillow](https://python-pillow.org/) | 纯 Python 视觉信息密度评分与 dHash 去重 | HPND |
| [Pydantic](https://docs.pydantic.dev/) & [Click](https://click.palletsprojects.com/) | 数据契约校验与 CLI 命令行入口 | MIT / BSD |
| [VidNoteK](https://github.com/WuHuiYang/VidNoteK) | 前置工程探索（继承并精简了字幕边界清洗与财经词表还原思路） | MIT |
