# FunASR 字幕生成器

将 WAV、MP3、MP4 等音视频文件转写为 SRT 或 VTT 字幕。支持音频预处理、FunASR VAD、说话人标签、日语文本清洗，以及可选的日中双语翻译。

## 功能

- 使用 FunASR/Qwen3 ASR 生成带时间轴的字幕。
- MP4 输入会先仅提取音频流为 WAV；不在提取阶段改变采样率或声道。
- 默认启用音频预处理、说话人识别和 FunASR `merge_vad`。
- 可选 DeepFilterNet2 降噪、Demucs 人声分离和长音频预切分。
- 可选 OpenAI 兼容接口的日语文本预处理与简体中文翻译。
- 默认将分段、OpenAI 请求内容和原始响应保存为本地 JSON 日志（不记录 API Key）；可用 `--no-local-log` 关闭。

## 安装

1. 安装与本机 CUDA 匹配的 PyTorch。

2. 安装核心 Python 依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 将 `ffmpeg` 和 `ffprobe` 加入 `PATH`。

4. 可选功能：

   - 降噪：准备项目目录中的 `deepFilter.exe`，或提供可执行文件路径。
   - 人声分离：安装 Demucs：`pip install demucs`。
   - 直接运行 `testQwen3.py`：安装 `qwen-asr`。

首次运行会下载 FunASR 模型。离线使用前，请先完整缓存 ASR、VAD、标点、说话人和可选强制对齐模型。

## 快速开始

```bash
python generate_subtitle.py input.wav
python generate_subtitle.py input.mp4 --format vtt
python generate_subtitle.py input.wav --bilingual
```

双语模式从环境变量读取密钥：

```powershell
$env:OPENAI_API_KEY = "your-key"
python generate_subtitle.py input.wav --bilingual
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--model` | ASR 模型，默认 `Qwen/Qwen3-ASR-1.7B`。 |
| `--device` | 推理设备，默认 `cuda:0`。 |
| `--offline` | 仅使用本地 Hugging Face 缓存，不访问网络。 |
| `--spk` / `--no-spk` | 开关说话人标签，默认开启。 |
| `--forced-aligner` | 为 Qwen3 开启强制对齐，默认关闭。 |
| `--preprocess-audio` / `--no-preprocess-audio` | 开关音频预处理，默认开启。 |
| `--enable-denoise` | 使用 DeepFilterNet2 降噪。 |
| `--enable-vocal-separation` | 使用 Demucs 提取人声。 |
| `--merge-vad` / `--no-merge-vad` | 开关 FunASR 内部 VAD 合并，默认开启。 |
| `--long-audio-split` | 在 ASR 前对超过阈值的音频做独立 VAD 切分，默认关闭。 |
| `--vad-split-threshold-seconds N` | 长音频切分阈值，单位秒。 |
| `--keep-audio-preprocess` | 保留 `.audio_preprocess` 中间文件。 |
| `--no-filter-fillers` | 保留纯填充词字幕段。 |
| `--bilingual` | 写入日语与简体中文双语字幕。 |
| `--preprocess-model` | 日语文本预处理模型。 |
| `--translation-model` | 最终翻译模型。 |
| `--preprocess-user-prompt` | 追加到预处理 user prompt 的指令。 |
| `--translation-user-prompt` | 追加到翻译 user prompt 的指令。 |
| `--translation-thinking` | 为 DeepSeek 最终翻译启用思考模式。 |

示例：

```bash
python generate_subtitle.py input.mp4 --no-spk --forced-aligner --bilingual --translation-thinking
python generate_subtitle.py input.wav --long-audio-split --vad-split-threshold-seconds 600 --keep-audio-preprocess
```

## 处理流程

```text
输入音视频
  → MP4 音频提取（仅 MP4）
  → 音频预处理
  → 可选长音频 VAD 分块
  → FunASR ASR / 标点 / 可选说话人识别
  → 字幕分段与填充词过滤
  → 可选文本预处理
  → 可选翻译
  → SRT / VTT 与 JSON 日志
```

## 输出文件

以 `input.srt` 为例：

| 文件 | 内容 |
|---|---|
| `input.srt` | 最终字幕。 |
| `input.srt.raw-segments.json` | ASR 原始字幕分段。 |
| `input.srt.segments.json` | 最终分段。 |
| `input.srt.openai-exchanges.json` | 双语流程的请求内容和 API 原始响应。 |
| `input.audio_preprocess/` | MP4 提取、预处理、VAD 分块等中间文件。默认在本次新建时自动删除。 |

`--keep-audio-preprocess` 可保留中间文件。为避免误删，运行前已存在的同名目录不会自动删除。
`--no-local-log` 会关闭上述 JSON 日志及预处理 `manifest.json`，不会影响 SRT/VTT 字幕输出。

## 安全与发布

- 不要将 `OPENAI_API_KEY`、`.env`、音视频、`.audio_preprocess/` 或 `*.openai-exchanges.json` 提交到 Git。
- API 交换日志包含字幕文本与模型原始输出，不含 Authorization 头或 API Key。
- 发布项目前应确认 FunASR、Qwen、DeepFilterNet2、Demucs 与各模型权重的许可证是否符合你的再分发方式。

## 测试

```bash
python -m unittest discover -s tests -v
```
