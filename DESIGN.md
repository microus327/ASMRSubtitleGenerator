# 字幕流水线设计

## 概览

这个工作区包含一套以 FunASR 为核心的字幕处理流水线：

1. [generate_subtitle.py](generate_subtitle.py) 负责统一编排 MP4 音频提取、字幕复用、ASR/VAD/说话人转写、基于 OpenAI 的文本清洗、翻译、字幕写出以及日志落地。
2. [preprocess_audio.py](preprocess_audio.py) 负责在 ASR 前执行可选的音频预处理。

当前设计将编排逻辑和较重的外部依赖调用保留在脚本层，同时把纯文本处理和字幕转换辅助逻辑尽量隔离出来，使其可以在不运行 FunASR 或 OpenAI 的情况下单独测试。

## 模块职责

### [generate_subtitle.py](generate_subtitle.py)

- 展开命令行输入路径并校验用户参数。
- 对 MP4 仅提取音频流为 WAV，不在提取阶段重采样或改变声道数。
- 在满足条件时复用已有字幕文件，跳过重复识别。
- 对每个输入文件执行 FunASR 转写；可选在转写前对长音频运行独立 VAD 分块。
- 从 ASR 结果构建标准化的字幕分段对象。
- 在需要时执行两轮 OpenAI 处理：
  - `text -> processed`：用于日语 ASR 结果后处理
  - `processed -> translation`：用于翻译为简体中文
- 输出字幕文件以及原始/最终分段的 JSON 日志。

### [preprocess_audio.py](preprocess_audio.py)

- 使用 `PreprocessOptions` 作为预处理配置边界。
- 根据目标输出路径构建默认预处理配置。
- 在独立运行预处理脚本时，用命令行参数覆盖默认配置。
- 执行音频预处理流水线，并写出记录步骤和产物的 `manifest.json`。

## 核心数据模型

字幕数据在函数之间通过字典传递，并遵循稳定的字段约定：

- `start`：起始时间，单位为毫秒
- `end`：结束时间，单位为毫秒
- `text`：本地清洗后的原始 ASR 文本
- `processed`：可选的 OpenAI 后处理日语文本
- `translation`：可选的简体中文翻译结果
- `spk`：可选的 FunASR 说话人编号

预期的数据流如下：

1. FunASR 产出 `text`
2. OpenAI 预处理产出 `processed`
3. OpenAI 翻译产出 `translation`

保留 `text` 字段，是为了让后续步骤可以对照原始识别、后处理结果以及翻译结果。

## 编排流程

对每一个输入文件，[generate_subtitle.py](generate_subtitle.py) 按以下顺序执行：

1. 解析输出字幕路径；MP4 输入先提取音频到 `<输出名>.audio_preprocess/00_extracted.wav`。
2. 如果未强制重新识别，则优先尝试复用已有字幕文件。
3. 按需调用 [preprocess_audio.py](preprocess_audio.py) 对源音频做预处理。
4. 如果启用了 `--long-audio-split` 且时长超过阈值，用 `fsmn-vad` 生成持久化分块。
5. 加载 FunASR 模型并执行转写；可选使用 `cam++` 生成说话人标签。
6. 将 ASR 输出转换为字幕分段字典。
7. 写出 `raw-segments` JSON，便于调试和审计。
8. 如果启用了双语模式：
   - 先用 OpenAI 对分段做预处理
   - 再用 OpenAI 对分段做翻译
   - 最后写出最终 `segments` JSON 和双语字幕
9. 如果未启用双语模式，则直接写出最终 `segments` JSON 和单语字幕。

## OpenAI 处理设计

`process_segments_with_openai(...)` 是预处理和翻译两条链路共享的变换引擎。

它负责：

- 按请求大小对分段进行分组
- 构造稳定的 JSON 载荷
- 调用兼容 OpenAI 的接口
- 将每次请求 payload 与 API 原始响应写入 `<字幕>.openai-exchanges.json`
- 从可能夹带额外包装文本的模型响应中提取 JSON
- 将每个返回分段重新映射回原始顺序
- 按字段语义应用回退策略

关键行为约束如下：

- 预处理阶段写入 `processed`，如果模型返回空值，则回退到原始文本。
- 翻译阶段写入 `translation`，不会在缺失输出时静默回退为原始日语文本。

## 音频预处理设计

[preprocess_audio.py](preprocess_audio.py) 采用线性流水线，步骤可选开启：

1. 将音频统一为单声道 / PCM16 WAV；默认输出 16kHz，但启用 DeepFilterNet2 时先转为 48kHz
2. 可选的 DeepFilterNet2 降噪；该步骤当前通过 `deepFilter input.wav --output-dir out_dir` 调用，且仅接受 48kHz WAV 输入
3. 如果启用了 DeepFilterNet2，则在降噪后再转回 16kHz 供后续 ASR 使用
4. 可选的去混响占位步骤
5. 可选的 Demucs 人声分离
6. 可选的峰值归一化或响度归一化

每次执行都会写出一个 `manifest.json`，其中记录：

- 原始输入路径
- 生效的预处理配置
- 按顺序执行的步骤
- 最终输出路径

这个 manifest 是排查预处理问题时最主要的调试产物。

## 日志与输出产物

当前流水线主要产出以下文件：

- 字幕文件：`.srt` 或 `.vtt`
- JSON 日志：`raw-segments` 和最终 `segments`
- 双语 API 交换日志：`openai-exchanges.json`
- 工作目录：`.audio_preprocess`，包含提取 WAV、预处理产物、manifest 和可选 VAD 分块。

JSON 日志的目的，是在不把大块模型输出直接刷到终端的前提下，保留可检查的中间结果。

`.audio_preprocess` 默认仅在本次运行创建时自动清理；指定 `--keep-audio-preprocess` 可保留。运行前已有的目录不会自动删除。

## 测试策略

当前的初始测试骨架集中覆盖 [tests/test_generate_subtitle.py](tests/test_generate_subtitle.py) 中的纯逻辑：

- 时间戳解析
- 从模型响应中提取 JSON
- 分段分组启发式
- ASR 结果到字幕分段的转换

这样可以让测试不依赖 FunASR、ffmpeg、OpenAI、DeepFilterNet2 和 Demucs。

## 扩展点

- 将去混响占位步骤替换为真实的模型实现。
- 如果终端日志和文件日志需要分离，可以把脚本层的 `print(...)` 迁移到结构化 logger。
- 如果 OpenAI 变换链继续扩张，可以把相关集成逻辑抽到独立模块。
- 在稳定样本音频和期望输出夹具准备好之后，补充端到端测试。
