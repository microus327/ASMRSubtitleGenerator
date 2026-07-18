# FunASR 字幕生成脚本。
#
# 用于从音频或视频生成 SRT/VTT 字幕。
#
# 示例：
# python generate_subtitle.py 厉害了.wav --bilingual --openai-url https://api.deepseek.com/chat/completions --openai-key sk-5bfe8de8ce2749f29ac31b0e1c6aaf67 --force-translation --force-recognition --disable-update --spk

from preprocess_audio import build_default_preprocess_options, preprocess_audio
from funasr import AutoModel
import requests
import glob
import time
import math
import json
import re
import os
import sys
import argparse
import logging
import subprocess
import tempfile
import shutil
import wave
from contextlib import contextmanager
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
)


def clean_text(text):
    return re.sub(r'<\|[^|]*\|>', '', text or "").strip()
 
model_short_names = {
    "funasr": "FunAudioLLM/Fun-ASR-Nano-2512",
    "qwen3": "Qwen/Qwen3-ASR-1.7B",
 }

CHINESE_FILLER_PATTERNS = [
    r"嗯",
    r"啊",
    r"呃",
    r"这个",
    r"那个",
    r"就是",
    r"然后",
    r"吧",
    r"嘛",
    r"呀",
    r"啦",
    r"好像",
    r"可能",
    r"其实",
    r"不过",
    r"那个",
]

JAPANESE_FILLER_PATTERNS = [
    r"あのね",
    r"えっと",
    r"えーと",
    r"えー",
    r"あー",
    r"うーん",
    r"うん",
    r"んー",
    r"ねえ",
    r"よね",
    r"よ",
    r"ね",
    r"さ",
    r"まあ",
    r"なんか",
    r"そのね",
    r"その",
    r"それで",
    r"でも",
    r"じゃあ",
    r"ほんとに",
    r"ほんと",
    r"まじ",
    r"ホント",
    r"は",
    r"はい"
    r"あ"
    r"し"
    r"う"
    r"ほ"
    r"ラ"
]

FILLER_PATTERNS = CHINESE_FILLER_PATTERNS + JAPANESE_FILLER_PATTERNS
FILLER_ONLY_RE = re.compile(r'^(?:' + '|'.join(FILLER_PATTERNS) + r')+$')

VAD_SPLIT_THRESHOLD_SECONDS = 600
VAD_SPLIT_THRESHOLD_MS = VAD_SPLIT_THRESHOLD_SECONDS * 1000

PREPROCESS_SYSTEM_PROMPT = """You are a professional Japanese ASR post-processing assistant.

Your task is to reconstruct and refine Japanese subtitle segments using the surrounding context before translation or further processing. Produce a natural, human-readable transcript while faithfully preserving the original meaning, speaking style, emotional tone, dialogue continuity, and ASMR atmosphere.

For each segment:

1. Remove meaningless filler expressions, interjections, filler-only segments, and obvious ASR artifacts when they do not contribute to the meaning, emotion, or atmosphere.

   Examples include isolated vocal sounds, accidental repetitions, and subtitle segments consisting entirely of such content.

   Within otherwise meaningful sentences, reduce excessive filler words or non-verbal vocalizations (such as あー, えー, うーん, あっ, んっ, はぁ, ふぅ, etc.) when they unnecessarily interrupt readability. Preserve those that clearly convey emotion, hesitation, rhythm, or the intended ASMR atmosphere.

   If an entire segment contains no meaningful content after processing, it may be omitted from the output.

2. Compress excessive repetitions.

   If the same word, phrase, or expression is intentionally repeated many times, preserve the intended emphasis while removing redundant repetitions. When appropriate, replace the omitted portion with an ellipsis (...) to maintain the original rhythm and speaking style.

3. Correct obvious ASR recognition errors.

   Use the surrounding context, grammar, and semantic consistency to identify words or phrases that are clearly incorrect or contradictory.

   Infer more appropriate replacements based on similar Japanese pronunciation (homophones or near-homophones), but only when the correction is highly confident. If multiple corrections are possible, choose the one requiring the smallest change to the original text.

   Never guess uncertain content.

Important rules:

- Preserve the original meaning whenever possible.
- Prioritize readability without changing the intended meaning.
- Do not summarize, paraphrase, or omit meaningful content.
- Do not invent information that is not supported by the context.
- Preserve speaker intent, emotional expression, dialogue flow, and ASMR atmosphere.
- Retain explicit, vulgar, erotic, or R18 language exactly as intended.
- Do not censor, sanitize, euphemize, or tone down explicit content.
- Preserve sentence boundaries whenever possible.
- Preserve proper names, numbers, and terminology unless they are clearly ASR recognition errors.
- Preserve the original segmentation whenever possible.
- A segment may be omitted only if it contains no meaningful content after processing.
- If no improvement is necessary, return the original text unchanged.

Preserve the original JSON structure and all existing fields. Modify only the value of the "processed" field.

Return only a valid JSON object with the following schema:

{"segments":[{"id":1,"processed":"..."}, ...]}
"""

TRANSLATION_SYSTEM_PROMPT = """You are a professional Japanese-to-Simplified-Chinese ASMR subtitle localization translator.

Your task is to translate the content of each segment's `processed` field into natural Simplified Chinese subtitles.

Before translating, treat all subtitle segments as parts of one continuous dialogue rather than independent sentences. Read the surrounding context to fully understand the conversation, then translate each segment accordingly while preserving the original segmentation.

Always use the `processed` field as the source text. If the original `text` field is also present, use it only as supplementary reference when resolving ambiguities. Never translate directly from the `text` field.

Japanese dialogue frequently omits subjects, objects, pronouns, and other grammatical elements. When the intended meaning is clear from the surrounding context, infer these omitted elements naturally so that the Chinese translation reads like authentic spoken dialogue rather than a literal translation.

Translate using both preceding and following segments to resolve:

- omitted subjects, objects, and pronouns
- unfinished sentences
- sentence fragments
- dialogue continuity
- emotional progression
- speaker intent
- references spanning multiple subtitle segments

Do not translate each subtitle segment in isolation. First understand the complete dialogue, then produce a coherent translation for every segment while preserving the original segmentation.

If a subtitle segment is incomplete or ambiguous by itself, infer its intended meaning from the surrounding context before translating. Produce a natural Chinese expression rather than a literal translation of the isolated fragment.

Produce subtitles that sound as if they were originally spoken in Chinese.

Prioritize:

- natural spoken Chinese
- dialogue fluency
- contextual consistency
- subtitle readability

over literal sentence-by-sentence translation, while faithfully preserving the original meaning, emotional tone, speaking style, pacing, and immersive ASMR atmosphere.

The source may contain R18 content including erotic dialogue, dirty talk, moaning, orgasm expressions, and explicit sexual descriptions.

Translate such content faithfully and naturally into Chinese. Preserve the original intensity, nuance, emotional expression, and speaking style.

Do not censor, euphemize, sanitize, omit, or rewrite explicit content.

When neighboring subtitle segments together form a complete expression, allow each translated segment to depend on its surrounding context while still matching its original segment.

Do not artificially preserve Japanese sentence structure if doing so produces unnatural Chinese.

If multiple translations are possible, choose the one that is most natural in the dialogue context.

Preserve the original JSON structure and all existing fields.

Do not add, remove, reorder, or modify any fields except the value of the "translation" field.

Return only a valid JSON object with the following schema:

{"segments":[{"id":1,"translation":"..."}, ...]}
"""

TRANSLATION_USER_PROMPT_PREFIX = (
    "Translate the following Japanese ASMR subtitle segments to Simplified Chinese. "
    "Use context across the supplied segments and keep the tone smooth. "
    "Keep the same segment order and return only JSON. "
)


def is_filler_only(text):
    if not text or not text.strip():
        return True
    cleaned = re.sub(r'[、。,.！？?!\s]+', '', text)
    if not cleaned:
        return True
    return bool(FILLER_ONLY_RE.fullmatch(cleaned))


def format_time_srt(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def format_time_vtt(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def parse_timestamp_ms(ts):
    if not ts:
        return 0
    ts = ts.strip().replace(".", ",", 1)
    parts = ts.split("-->" if "-->" in ts else " ")
    if len(parts) == 1:
        segment = parts[0].strip()
    else:
        segment = parts[0].strip()
    parts = segment.split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
        if "," in s_ms:
            s, ms = s_ms.split(",")
        else:
            s, ms = s_ms.split(".") if "." in s_ms else (s_ms, "0")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms.ljust(3, "0")[:3])
    return 0


def load_existing_subtitles(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    segments = []
    idx = 0
    if lines and lines[0].startswith("WEBVTT"):
        while idx < len(lines) and lines[idx].strip():
            idx += 1
        idx += 1

    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        if lines[idx].strip().isdigit():
            idx += 1
            continue
        if "-->" not in lines[idx]:
            idx += 1
            continue
        time_line = lines[idx].strip()
        start_str, end_str = [part.strip()
                              for part in time_line.split("-->", 1)]
        start = parse_timestamp_ms(start_str)
        end = parse_timestamp_ms(end_str)
        idx += 1
        text_lines = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        if not text_lines:
            continue
        source = text_lines[0]
        translation = "\n".join(text_lines[1:]) if len(text_lines) > 1 else ""
        segments.append({"start": start, "end": end,
                        "text": source, "translation": translation})
    return segments


def timestamp_bounds_ms(result):
    bounds = []
    for key in ("timestamp", "timestamps"):
        for ts in result.get(key, []) or []:
            if isinstance(ts, dict):
                start = ts.get("start_time", ts.get("start"))
                end = ts.get("end_time", ts.get("end"))
                if start is None or end is None:
                    continue
                start_ms = int(float(start) * 1000)
                end_ms = int(float(end) * 1000)
            elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start_ms = int(ts[0])
                end_ms = int(ts[1])
            else:
                continue
            if end_ms > start_ms:
                bounds.append((start_ms, end_ms))
    if not bounds:
        return None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def split_segments_for_translation(segments, max_tokens=512):
    # 按上下文窗口对字幕分段分组，供 OpenAI 分批处理。
    if max_tokens <= 0:
        return [segments]
    groups = []
    current = []
    current_tokens = 0

    def tokens_of(text):
        # 使用基于字符数的简化估算，避免单次请求超过上下文预算。
        return max(1, math.ceil(len(text) / 4))

    for seg in segments:
        seg_tokens = tokens_of(seg["text"])
        if current and current_tokens + seg_tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        groups.append(current)
    return groups


def build_openai_payload(segment_group):
    return {
        "segments": [
            {
                "id": i,
                "text": seg["text"],
                "processed": seg.get("processed"),
                "speaker": seg.get("spk"),
            }
            for i, seg in enumerate(segment_group, 1)
        ],
    }


def process_segments_with_openai(api_key, model, segments, max_context_tokens, *,
                                 openai_base=None, verbose=True, max_retries=3,
                                 action_label="Processing", result_field="text",
                                 system_prompt="", user_prompt_prefix="",
                                 fallback_to_original_text=True, temperature=0):
    # 对字幕分段执行一轮 OpenAI 变换处理。
    api_url = openai_base or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    processed_segments = []
    groups = split_segments_for_translation(segments, max_context_tokens)
    total = len(groups)

    for idx, group in enumerate(groups, 1):
        payload = build_openai_payload(group)
        if verbose:
            print(f"{action_label} chunk {idx}/{total} ({len(group)} segments) ...")

        user_prompt = (
            f"{user_prompt_prefix}"
            f"Input: {json.dumps(payload, ensure_ascii=False)}")

        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature
        }

        success = False
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    api_url, headers=headers, json=request_payload, timeout=120)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenAI {action_label.lower()} failed: {response.status_code} {response.text}")

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                data = parse_json_from_text(content)
                output_segment_map = {
                    seg.get("id"): seg.get(result_field, "")
                    for seg in data.get("segments", [])
                    if seg.get("id") is not None
                }
                for i, seg in enumerate(group, 1):
                    output_text = output_segment_map.get(i, "")
                    fallback_text = seg.get("processed") or seg["text"]
                    processed_segments.append({
                        **seg,
                        result_field: output_text if output_text or not fallback_to_original_text else fallback_text,
                    })
                success = True
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    if verbose:
                        print(
                            f"{action_label} attempt {attempt}/{max_retries} failed: {exc}. Retrying...")
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"OpenAI {action_label.lower()} failed after {max_retries} attempts: {exc}") from exc

        if not success:
            raise RuntimeError(
                f"OpenAI {action_label.lower()} failed after {max_retries} attempts: {last_error}")

    return processed_segments


def preprocess_segments_with_openai(api_key, model, segments, max_context_tokens, openai_base, verbose=True, max_retries=3):
    # 在保持分段顺序不变的前提下，先清洗日语 ASR 文本再进入翻译。
    return process_segments_with_openai(
        api_key,
        model,
        segments,
        max_context_tokens,
        openai_base=openai_base,
        verbose=verbose,
        max_retries=max_retries,
        action_label="Preprocessing",
        result_field="processed",
        system_prompt=PREPROCESS_SYSTEM_PROMPT,
        user_prompt_prefix="Keep the same segment order and return only JSON. ",
        fallback_to_original_text=True,
    )


def parse_json_from_text(text):
    # 从可能带有包裹文本的模型响应中提取 JSON 对象。
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise


def expand_input_paths(patterns):
    # 支持从多个输入参数和通配符中展开可用音视频文件
    expanded = []
    for pattern in patterns:
        matches = [match for match in glob.glob(
            pattern, recursive=True) if os.path.isfile(match)]
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    unique_paths = []
    seen = set()
    for path in expanded:
        normalized = os.path.normpath(path)
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(path)
    return unique_paths


def build_model_kwargs(args):
    # 根据命令行参数构造 FunASR 模型初始化参数。
    kwargs = {
        "model": args.model,
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "punc_model": "ct-punc",
        "device": args.device,
        "disable_update": args.disable_update,
    }
    if args.spk:
        kwargs["spk_model"] = "cam++"
    if "Fun-ASR-Nano" in args.model or "Qwen" in args.model:
        kwargs["trust_remote_code"] = True
        kwargs["hub"] = "hf"
    # if "Qwen3" in args.model:
    #     kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
    return kwargs


def build_generate_kwargs(input_path, args):
    # 为单个输入文件构造 FunASR 推理参数。
    language = "Japanese" if "Qwen3" in args.model else "日文"
    generate_kwargs = {
        "input": input_path,
        "batch_size": 1,
        "batch_size_s": 300,
        "sentence_timestamp": True,
        "output_timestamp": True,
        "return_time_stamps": True,
        "merge_vad": True
    }
    generate_kwargs["language"] = language
    return generate_kwargs


def build_preprocess_options(args, output_path):
    return build_default_preprocess_options(args.preprocess_audio, output_path)


def build_segments_from_result(result_item, filter_fillers):
    # 从单个 FunASR 结果项中整理出标准字幕分段。
    segments = []
    for seg in result_item.get("sentence_info", []) or []:
        text = clean_text(seg.get("sentence") or seg.get("text", ""))
        if filter_fillers and is_filler_only(text):
            continue
        start = int(seg.get("start", 0) or 0)
        end = int(seg.get("end", 0) or 0)
        if text and end > start:
            segments.append({"start": start, "end": end,
                            "text": text, "spk": seg.get("spk")})

    if not segments:
        text = clean_text(result_item.get("text", ""))
        if text:
            start, end = timestamp_bounds_ms(result_item) or (0, 0)
            segments.append({"start": start, "end": end,
                            "text": text, "spk": None})
    return segments


def write_subtitles(output_path, segments, output_format, bilingual, include_speaker=False):
    # 将字幕分段写入 SRT 或 VTT 文件。
    fmt = format_time_srt if output_format == "srt" else format_time_vtt
    if bilingual:
        write_bilingual_srt(output_path, segments, fmt, output_format,
                            include_speaker=include_speaker)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        if output_format == "vtt":
            f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            print(f"[Subtitle] Writing segment {i}: {seg['processed']}ms")
            text = f"[Speaker {seg['spk']}] {seg['processed']}" if include_speaker and seg.get(
                "spk") is not None else seg["processed"]
            if output_format == "srt":
                f.write(
                    f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")
            else:
                f.write(
                    f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")


def translate_segments_with_openai(api_key, model, segments, max_context_tokens, openai_base=None, verbose=True, max_retries=3):
    # 将预处理后的日语字幕分段翻译成简体中文。
    processed_segments = process_segments_with_openai(
        api_key,
        model,
        segments,
        max_context_tokens,
        openai_base=openai_base,
        verbose=verbose,
        max_retries=max_retries,
        action_label="Translation",
        result_field="translation",
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        user_prompt_prefix=TRANSLATION_USER_PROMPT_PREFIX,
        fallback_to_original_text=False,
    )
    return processed_segments


def write_bilingual_srt(output_path, segments, fmt, output_format, include_speaker=False):
    # 将识别结果和翻译结果写成双语字幕文件
    with open(output_path, "w", encoding="utf-8") as f:
        if output_format == "vtt":
            f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            source_text = f"[Speaker {seg['spk']}] {seg['processed']}" if include_speaker and seg.get(
                "spk") is not None else seg['processed']
            target_text = seg.get("translation") or ""
            f.write(
                f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{source_text}\n{target_text}\n\n")


def write_segments_log(output_path, segments, stage="segments"):
    log_path = f"{output_path}.{stage}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segments log saved to: {log_path}")


def print_completion(output_path, segments, start_time=None):
    # 输出统一的完成摘要，避免各分支重复拼装日志。
    print(f"Done! {len(segments)} subtitles → {output_path}")
    if start_time is not None:
        total_duration = time.time() - start_time
        print(f"Total elapsed time: {total_duration:.2f}s")


def run_bilingual_pipeline(segments, args, openai_key, openai_base, output_path, start_time):
    # 执行双语链路：先预处理文本，再翻译，最后写出双语字幕。
    print("Preprocessing segments with OpenAI before translation...")
    preprocess_start = time.time()
    segments = preprocess_segments_with_openai(
        openai_key,
        args.openai_model,
        segments,
        args.max_context_tokens,
        openai_base=openai_base,
        verbose=True,
        max_retries=3,
    )
    preprocess_duration = time.time() - preprocess_start
    print(f"Preprocessing completed in {preprocess_duration:.2f}s")

    print("Translating segments with OpenAI...")
    translation_start = time.time()
    segments = translate_segments_with_openai(
        openai_key,
        args.openai_model,
        segments,
        args.max_context_tokens,
        openai_base=openai_base,
        verbose=True,
        max_retries=3,
    )
    translation_duration = time.time() - translation_start
    print(f"Translation completed in {translation_duration:.2f}s")

    write_segments_log(output_path, segments)
    write_subtitles(output_path, segments, args.format,
                    args.bilingual, include_speaker=args.spk)
    print_completion(output_path, segments, start_time)
    return segments


def try_reuse_existing_subtitles(output_path, args, openai_key, openai_base, start_time):
    # 在允许跳过识别时，优先复用已有字幕文件。
    if not os.path.exists(output_path) or args.force_recognition:
        return False

    print(
        f"Found existing subtitle file {output_path}; loading and checking reuse conditions.")
    segments = load_existing_subtitles(output_path)
    if not segments:
        print("Failed to parse existing subtitle file or file is empty.")
        sys.exit(1)

    if args.bilingual:
        need_translation = args.force_translation or any(
            not seg.get("translation") for seg in segments)
        if need_translation:
            if args.force_translation:
                print("Existing subtitle loaded; force re-translation enabled.")
            else:
                print(
                    "Existing subtitle loaded; translating missing bilingual lines.")
            run_bilingual_pipeline(
                segments, args, openai_key, openai_base, output_path, start_time)
            return True

        print(
            "Existing bilingual subtitle loaded; skipping recognition and translation.")
        print_completion(output_path, segments)
        return True

    print("Existing subtitle loaded; skipping recognition.")
    print_completion(output_path, segments)
    return True


def probe_audio_duration_ms(input_path):
    """Return the media duration in milliseconds, or ``None`` if it cannot be read."""
    try:
        with wave.open(input_path, "rb") as wav_file:
            return int(wav_file.getnframes() * 1000 / wav_file.getframerate())
    except (wave.Error, EOFError, FileNotFoundError):
        pass

    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path,
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return int(float(completed.stdout.strip()) * 1000)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def extract_vad_intervals(vad_result):
    """Normalize FunASR VAD's nested ``value`` field into millisecond intervals."""
    value = vad_result[0].get("value", []) if vad_result else []
    while len(value) == 1 and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        value = value[0]

    intervals = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    return intervals


def group_vad_intervals(intervals, max_duration_ms=VAD_SPLIT_THRESHOLD_MS):
    """Group adjacent speech regions into chunks no longer than the configured limit."""
    groups = []
    chunk_start = chunk_end = None
    for start, end in sorted(intervals):
        if chunk_start is None:
            chunk_start, chunk_end = start, end
        elif end - chunk_start <= max_duration_ms:
            chunk_end = end
        else:
            groups.append((chunk_start, chunk_end))
            chunk_start, chunk_end = start, end
    if chunk_start is not None:
        groups.append((chunk_start, chunk_end))
    return groups


def write_audio_chunk(input_path, output_path, start_ms, end_ms):
    command = [
        "ffmpeg", "-y", "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
        "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", output_path,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


@contextmanager
def split_long_audio_with_vad(input_path, args):
    """Yield ``(path, original_offset_ms)`` inputs, VAD-splitting media over 600 seconds."""
    duration_ms = probe_audio_duration_ms(input_path)
    if duration_ms is None:
        print("[VAD] Could not determine input duration; skipping pre-transcription splitting.")
        yield [(input_path, 0)]
        return
    if duration_ms <= VAD_SPLIT_THRESHOLD_MS:
        print(
            f"[VAD] Input duration: {duration_ms / 1000:.1f}s "
            f"(threshold: {VAD_SPLIT_THRESHOLD_SECONDS}s); no pre-split needed.")
        yield [(input_path, 0)]
        return

    print(
        f"[VAD] Input duration: {duration_ms / 1000:.1f}s "
        f"(threshold: {VAD_SPLIT_THRESHOLD_SECONDS}s); running FunASR VAD...")
    vad_start = time.time()
    vad_model = AutoModel(
        model="fsmn-vad", device=args.device, disable_update=args.disable_update)
    vad_result = vad_model.generate(input=input_path, batch_size_s=300)
    intervals = extract_vad_intervals(vad_result)
    chunks = group_vad_intervals(intervals)
    print(
        f"[VAD] Detection completed in {time.time() - vad_start:.2f}s: "
        f"{len(intervals)} speech interval(s), grouped into {len(chunks)} chunk(s).")
    if not chunks:
        print("[VAD] No speech segments found; transcribing the original input.")
        yield [(input_path, 0)]
        return

    temp_dir = tempfile.mkdtemp(prefix="funasr-vad-")
    try:
        chunk_inputs = []
        for index, (start_ms, end_ms) in enumerate(chunks, 1):
            chunk_path = os.path.join(temp_dir, f"part_{index:04d}.wav")
            print(
                f"[VAD] Exporting chunk {index}/{len(chunks)}: "
                f"{start_ms / 1000:.3f}s - {end_ms / 1000:.3f}s")
            write_audio_chunk(input_path, chunk_path, start_ms, end_ms)
            chunk_inputs.append((chunk_path, start_ms))
        print(f"[VAD] Created {len(chunk_inputs)} temporary transcription chunk(s).")
        yield chunk_inputs
    finally:
        if args.keep_vad_chunks:
            print(f"[VAD] Keeping temporary transcription chunks: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_transcription(input_path, args):
    # 加载 ASR 模型并转写单个输入文件或预先切分的音频块。
    kwargs = build_model_kwargs(args)
    print("Loading model...")
    model_load_start = time.time()
    model = AutoModel(**kwargs)

    model_load_duration = time.time() - model_load_start
    print(f"Model loaded in {model_load_duration:.2f}s")
    print("Transcribing...")
    transcribe_start = time.time()
    chunk_inputs = input_path if isinstance(input_path, list) else [(input_path, 0)]
    sentence_info = []
    texts = []
    for index, (chunk_path, offset_ms) in enumerate(chunk_inputs, 1):
        if len(chunk_inputs) > 1:
            print(f"[ASR] Transcribing VAD chunk {index}/{len(chunk_inputs)} (offset: {offset_ms}ms)...")
        result = model.generate(**build_generate_kwargs(chunk_path, args))
        result_item = result[0]
        texts.append(result_item.get("text", ""))
        chunk_sentences = result_item.get("sentence_info", []) or []
        if chunk_sentences:
            for sentence in chunk_sentences:
                adjusted = dict(sentence)
                adjusted["start"] = int(adjusted.get("start", 0) or 0) + offset_ms
                adjusted["end"] = int(adjusted.get("end", 0) or 0) + offset_ms
                sentence_info.append(adjusted)
        elif result_item.get("text"):
            start, end = timestamp_bounds_ms(result_item) or (0, 0)
            sentence_info.append({
                "sentence": result_item["text"],
                "start": start + offset_ms,
                "end": end + offset_ms,
            })
    transcribe_duration = time.time() - transcribe_start
    print(f"Transcription completed in {transcribe_duration:.2f}s")
    return {"text": "\n".join(texts), "sentence_info": sentence_info}


def write_recognition_output(output_path, segments, args, openai_key, openai_base, start_time):
    # 将识别结果写入日志和最终字幕文件。
    write_segments_log(output_path, segments, stage="raw-segments")

    if args.bilingual:
        run_bilingual_pipeline(segments, args, openai_key,
                               openai_base, output_path, start_time)
        return

    write_segments_log(output_path, segments, stage="segments")
    write_subtitles(output_path, segments, args.format,
                    args.bilingual, include_speaker=args.spk)
    print_completion(output_path, segments, start_time)


def process_input_file(input_path, args, openai_key, openai_base, start_time):
    # 处理单个输入文件，覆盖字幕复用、ASR 和可选翻译链路。
    output_path = args.output or f"{os.path.splitext(input_path)[0]}.{args.format}"
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    if try_reuse_existing_subtitles(output_path, args, openai_key, openai_base, start_time):
        return

    effective_input_path = input_path
    preprocess_options = build_preprocess_options(args, output_path)
    if preprocess_options.enabled:
        print("Preprocessing audio...")
        preprocess_start = time.time()
        effective_input_path = preprocess_audio(input_path, preprocess_options)
        preprocess_duration = time.time() - preprocess_start
        print(f"Audio preprocessing completed in {preprocess_duration:.2f}s")
        print(f"Preprocessed audio: {effective_input_path}")

    with split_long_audio_with_vad(effective_input_path, args) as transcription_inputs:
        result_item = run_transcription(transcription_inputs, args)
    segments = build_segments_from_result(result_item, args.filter_fillers)
    if not segments:
        print("No speech detected.")
        return

    write_recognition_output(output_path, segments, args,
                             openai_key, openai_base, start_time)


def main():
    parser = argparse.ArgumentParser(
        description="Generate subtitles from audio/video using FunASR")
    parser.add_argument("input", nargs="+",
                        help="Audio/video file paths or glob patterns")
    parser.add_argument(
        "-o", "--output", help="Output file (default: input.srt)")
    parser.add_argument("--format", choices=["srt", "vtt"], default="srt")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-update", action="store_true",
                        help="Skip FunASR model update/check when loading the model")
    parser.add_argument("--spk", action="store_true",
                        help="Include speaker labels")
    parser.add_argument("--no-filter-fillers", dest="filter_fillers",
                        action="store_false", help="Do not remove filler-only segments")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument(
        "--openai-url", help="OpenAI API base URL, e.g. https://api.gptsapi.net/v1")
    parser.add_argument("--openai-model", default="deepseek-v4-flash",
                        help="OpenAI model name for translation")
    parser.add_argument("--max-context-tokens", type=int, default=512,
                        help="Max context size for OpenAI translation upload")
    parser.add_argument("--bilingual", action="store_true",
                        help="Write bilingual subtitles with Japanese above Chinese")
    parser.add_argument("--force", "--force-recognition", dest="force_recognition", action="store_true",
                        help="Force re-recognition even if same-name subtitle file exists")
    parser.add_argument("--force-translation", dest="force_translation", action="store_true",
                        help="Force re-translation of bilingual subtitles even if existing translations are present")
    parser.add_argument("--preprocess-audio", action="store_true",
                        help="Enable audio preprocessing before ASR")
    parser.add_argument("--keep-vad-chunks", action="store_true",
                        help="Keep VAD-split temporary WAV files instead of deleting them")
    args = parser.parse_args()

    input_paths = expand_input_paths(args.input)
    if not input_paths:
        print("Error: no input files matched")
        sys.exit(1)
    if args.output and len(input_paths) > 1:
        print("Error: --output cannot be used when multiple input files are provided")
        sys.exit(1)

    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if args.bilingual and not openai_key:
        print("Error: bilingual mode requires --openai-key or OPENAI_API_KEY environment variable")
        sys.exit(1)
    openai_base = args.openai_url.rstrip("/") if args.openai_url else None

    if model_short_names.get(args.model):
        args.model = model_short_names[args.model]

    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            sys.exit(1)
        process_input_file(input_path, args, openai_key,
                           openai_base, time.time())


if __name__ == "__main__":
    main()

# 待办：1.VAD对Qwen貌似不生效 2.说活人分离 3.Preprocessing chunk 1/1 (88 segments) ... 分chunk似乎也没生效
