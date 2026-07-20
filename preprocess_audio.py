import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_NORMALIZE_MODE = "loudness|peak"
SUPPORTED_NORMALIZE_STEPS = {"peak", "loudness"}
DEEPFILTERNET2_SAMPLE_RATE = 48000


# 音频预处理流水线的配置对象。
@dataclass
class PreprocessOptions:
    enabled: bool = False
    work_dir: str | None = None
    ffmpeg_bin: str = "ffmpeg"
    enable_denoise: bool = False
    deepfilter_bin: str = ".\\deepFilter"
    deepfilter_model: str | None = None
    deepfilter_post_filter: bool = True
    deepfilter_post_filter_beta: float = 0.02
    deepfilter_atten_lim_db: float = 30.0
    deepfilter_compensate_delay: bool = True
    enable_dereverb: bool = False
    enable_vocal_separation: bool = False
    demucs_model: str = "htdemucs"
    normalize: str = DEFAULT_NORMALIZE_MODE
    peak_target_db: float = -1.0
    loudness_target_lufs: float = -16.0
    keep_intermediate: bool = True
    write_manifest: bool = True
    sample_rate: int = 16000
    channels: int = 1


def build_default_preprocess_options(enabled, output_path):
    # 根据目标输出路径构造默认的音频预处理配置。
    work_dir = f"{os.path.splitext(output_path)[0]}.audio_preprocess"
    return PreprocessOptions(enabled=enabled, work_dir=work_dir)


def apply_cli_overrides(options, args):
    # 用命令行参数覆盖默认的预处理配置。
    options.work_dir = args.work_dir or options.work_dir
    options.ffmpeg_bin = args.ffmpeg_bin
    options.enable_denoise = args.denoise
    options.deepfilter_bin = args.deepfilter_bin
    options.deepfilter_model = args.deepfilter_model
    options.deepfilter_post_filter = args.deepfilter_post_filter
    options.deepfilter_post_filter_beta = args.deepfilter_post_filter_beta
    options.deepfilter_atten_lim_db = args.deepfilter_atten_lim_db
    options.deepfilter_compensate_delay = args.deepfilter_compensate_delay
    options.enable_dereverb = args.dereverb
    options.enable_vocal_separation = args.vocal_separation
    options.demucs_model = args.demucs_model
    options.normalize = args.normalize
    options.peak_target_db = args.peak_target_db
    options.loudness_target_lufs = args.loudness_target_lufs
    options.keep_intermediate = args.keep_intermediate
    return options


def _parse_normalize_mode(normalize_mode):
    # 将归一化配置解析成可顺序执行的步骤列表，支持 loudness|peak 这类链式写法。
    if not normalize_mode or normalize_mode == "none":
        return []

    raw_steps = [step.strip().lower()
                 for step in normalize_mode.split("|") if step.strip()]
    if not raw_steps:
        return []
    if "none" in raw_steps:
        raise ValueError(
            "normalize mode 'none' cannot be combined with other modes")

    parsed_steps = []
    for step in raw_steps:
        if step not in SUPPORTED_NORMALIZE_STEPS:
            supported = "|".join(sorted(SUPPORTED_NORMALIZE_STEPS))
            raise ValueError(
                f"unsupported normalize mode: {step}. Supported values: none, peak, loudness, {supported}")
        if step not in parsed_steps:
            parsed_steps.append(step)
    return parsed_steps


def _run_command(command, step_name):
    # 执行外部命令，并在失败时抛出可定位的问题信息。
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{step_name} failed: command not found: {command[0]}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"{step_name} failed: {details}")
    return result


def _find_single_audio_file(directory):
    candidates = sorted(Path(directory).rglob("*.wav"))
    if not candidates:
        raise RuntimeError(f"No WAV output found in {directory}")
    return candidates[0]


def _write_manifest(work_dir, manifest):
    # 持久化预处理输入、步骤和输出的机器可读记录。
    manifest_path = Path(work_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _read_wav_format(input_path):
    input_path = Path(input_path)
    try:
        with wave.open(str(input_path), "rb") as wav_file:
            return {
                "sample_rate": wav_file.getframerate(),
                "channels": wav_file.getnchannels(),
            }
    except (wave.Error, FileNotFoundError, EOFError):
        return None


def _copy_wav_file(input_path, output_path):
    shutil.copy2(input_path, output_path)
    return output_path


def _standardize_audio(input_path, output_path, options, sample_rate=None):
    target_sample_rate = sample_rate if sample_rate is not None else options.sample_rate
    wav_format = _read_wav_format(input_path)
    need_resample = wav_format is None or wav_format["sample_rate"] != target_sample_rate
    need_channel_convert = wav_format is None or wav_format["channels"] != options.channels
    if not need_resample and not need_channel_convert:
        return _copy_wav_file(input_path, output_path)
    command = [options.ffmpeg_bin, "-y", "-i", str(input_path)]
    if need_resample:
        command.extend([
            "-ar",
            str(target_sample_rate)
        ])
    if need_channel_convert:
        command.extend([
            "-ac",
            str(options.channels)
        ])
    command.append(str(output_path))
    _run_command(command, "Audio resampling and channel conversion")
    return output_path


def _apply_deepfilternet2(input_path, output_path, options):
    output_dir = output_path.parent / "deepfilternet2"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [options.deepfilter_bin]
    if options.deepfilter_model:
        command.extend(["-m", options.deepfilter_model])
    if options.deepfilter_post_filter:
        command.append("--pf")
        command.extend(["--pf-beta", str(options.deepfilter_post_filter_beta)])
    command.extend(["-a", str(options.deepfilter_atten_lim_db)])
    if options.deepfilter_compensate_delay:
        command.append("-D")
    command.extend([
        str(input_path),
        "--output-dir",
        str(output_dir),
    ])
    _run_command(command, "DeepFilterNet2 denoise")
    produced = output_dir / input_path.name
    if not produced.exists():
        produced = _find_single_audio_file(output_dir)
    shutil.copy2(produced, output_path)
    return output_path


def _apply_dereverb_placeholder(input_path, output_path):
    print("Dereverberation is enabled, but currently uses a passthrough placeholder.")
    shutil.copy2(input_path, output_path)
    return output_path


def _apply_demucs(input_path, output_path, options):
    output_dir = output_path.parent / "demucs"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        options.demucs_model,
        "-o",
        str(output_dir),
        str(input_path),
    ]
    _run_command(command, "Demucs vocal separation")
    vocals_path = output_dir / options.demucs_model / input_path.stem / "vocals.wav"
    if not vocals_path.exists():
        raise RuntimeError(f"Demucs vocal stem not found: {vocals_path}")
    shutil.copy2(vocals_path, output_path)
    return output_path


def _detect_peak_db(input_path, ffmpeg_bin):
    null_device = "NUL" if os.name == "nt" else "/dev/null"
    command = [
        ffmpeg_bin,
        "-i",
        str(input_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        null_device,
    ]
    result = _run_command(command, "Peak detection")
    combined = "\n".join([result.stdout or "", result.stderr or ""])
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", combined)
    return float(match.group(1)) if match else None


def _normalize_peak(input_path, output_path, options):
    peak_db = _detect_peak_db(input_path, options.ffmpeg_bin)
    if peak_db is None:
        print("Peak detection unavailable; skipping peak normalization.")
        shutil.copy2(input_path, output_path)
        return output_path

    gain_db = options.peak_target_db - peak_db
    command = [
        options.ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        f"volume={gain_db:.3f}dB",
        str(output_path),
    ]
    _run_command(command, "Peak normalization")
    return output_path


def _normalize_loudness(input_path, output_path, options):
    command = [
        options.ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        f"loudnorm=I={options.loudness_target_lufs}:TP={options.peak_target_db}:LRA=11",
        str(output_path),
    ]
    _run_command(command, "Loudness normalization")
    return output_path


def preprocess_audio(input_path, options: PreprocessOptions):
    # 执行已配置的预处理步骤，并返回最终音频路径。
    if not options.enabled:
        return input_path

    input_path = Path(input_path)
    work_dir = Path(options.work_dir) if options.work_dir else input_path.parent / \
        f"{input_path.stem}.audio_preprocess"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": str(input_path),
        "options": asdict(options),
        "steps": [],
    }
    intermediate_paths = []
    normalize_steps = _parse_normalize_mode(options.normalize)

    total_pipeline_steps = 1
    if options.enable_denoise:
        total_pipeline_steps += 2
    if options.enable_dereverb:
        total_pipeline_steps += 1
    if options.enable_vocal_separation:
        total_pipeline_steps += 1
    total_pipeline_steps += len(normalize_steps) or 1
    current_pipeline_step = 1

    # 1) 统一采样率、声道数和编码格式，保证后续处理链输入稳定。
    standardized_path = work_dir / "01_standardized.wav"
    standardize_sample_rate = DEEPFILTERNET2_SAMPLE_RATE if options.enable_denoise else options.sample_rate
    print(
        f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: "
        f"standardize to {standardize_sample_rate / 1000:.0f}kHz / mono"
    )
    current_path = _standardize_audio(
        input_path, standardized_path, options, sample_rate=standardize_sample_rate)
    manifest["steps"].append(
        {"step": "standardize", "output": str(current_path)})
    if Path(current_path) != input_path:
        intermediate_paths.append(current_path)
    current_pipeline_step += 1

    # 2) 标准化后立即做归一化，支持 loudness|peak 这种串联模式。
    if not normalize_steps:
        manifest["steps"].append(
            {"step": "normalize_skipped", "output": str(current_path)})
        current_pipeline_step += 1
    else:
        final_normalize_path = work_dir / "02_normalize.wav"  
        total_normalize_steps = len(normalize_steps)
        for step_index, normalize_step in enumerate(normalize_steps, 1):
            is_last_normalize_step = step_index == total_normalize_steps
            has_following_processing = (
                options.enable_denoise
                or options.enable_dereverb
                or options.enable_vocal_separation
            )
            if is_last_normalize_step and not has_following_processing:
                step_output_path = final_normalize_path
            else:
                step_output_path = work_dir / f"02_{step_index:02d}_{normalize_step}_normalized.wav"

            if normalize_step == "loudness":
                print(
                    f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: loudness normalization")
                current_path = _normalize_loudness(
                    current_path, step_output_path, options)
                manifest["steps"].append(
                    {"step": "loudness_normalize", "output": str(current_path)})
            else:
                print(
                    f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: peak normalization")
                current_path = _normalize_peak(
                    current_path, step_output_path, options)
                manifest["steps"].append(
                    {"step": "peak_normalize", "output": str(current_path)})

            intermediate_paths.append(current_path)
            current_pipeline_step += 1

    # 3) 可选降噪，默认按既定方案接 DeepFilterNet2。
    if options.enable_denoise:
        denoise_path = work_dir / "03_denoised.wav"
        print(
            f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: denoise with DeepFilterNet2")
        current_path = _apply_deepfilternet2(
            current_path, denoise_path, options)
        manifest["steps"].append(
            {"step": "denoise", "output": str(current_path)})
        intermediate_paths.append(current_path)
        current_pipeline_step += 1

        resampled_path = work_dir / "04_resampled_for_asr.wav"
        print(
            f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: "
            f"resample denoised audio to {options.sample_rate / 1000:.0f}kHz for ASR"
        )
        current_path = _standardize_audio(
            current_path, resampled_path, options)
        manifest["steps"].append(
            {"step": "resample_for_asr", "output": str(current_path)})
        intermediate_paths.append(current_path)
        current_pipeline_step += 1

    # 4) 去混响先保留流程位置，当前实现为透传占位，方便后续替换真实模型。
    if options.enable_dereverb:
        dereverb_path = work_dir / "05_dereverb.wav"
        print(
            f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: dereverberation placeholder")
        current_path = _apply_dereverb_placeholder(current_path, dereverb_path)
        manifest["steps"].append(
            {"step": "dereverb_placeholder", "output": str(current_path)})
        intermediate_paths.append(current_path)
        current_pipeline_step += 1

    # 5) 背景音乐明显时可选做人声分离，当前保留 Demucs 的 vocals stem。
    if options.enable_vocal_separation:
        vocals_path = work_dir / "06_vocals.wav"
        print(
            f"[AudioPreprocess] Step {current_pipeline_step}/{total_pipeline_steps}: vocal separation with Demucs")
        current_path = _apply_demucs(current_path, vocals_path, options)
        manifest["steps"].append(
            {"step": "vocal_separation", "output": str(current_path)})
        intermediate_paths.append(current_path)
        current_pipeline_step += 1

    manifest["final_output"] = str(current_path)
    if options.write_manifest:
        _write_manifest(work_dir, manifest)

    # for path in intermediate_paths:
    #     print(f"[AudioPreprocess] Intermediate output: {path}")
    if not options.keep_intermediate:
        for path in intermediate_paths:
            if Path(path) != Path(current_path) and Path(path).exists():
                Path(path).unlink()

    print(f"[AudioPreprocess] Final output: {current_path}")
    return str(current_path)


def main():
    # 独立运行音频预处理脚本时的命令行入口。
    parser = argparse.ArgumentParser(
        description="Audio preprocessing pipeline for ASR input")
    parser.add_argument("input", help="Input audio path")
    parser.add_argument(
        "--work-dir", help="Directory for intermediate and final files")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable DeepFilterNet2 denoise")
    parser.add_argument(
        "--deepfilter-bin",
        default="deepFilter",
        help="DeepFilterNet2 CLI command name or path",
    )
    parser.add_argument("--deepfilter-model",
                        help="Path to DeepFilterNet model tar.gz")
    parser.add_argument("--deepfilter-post-filter", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable DeepFilter post-filter")
    parser.add_argument("--deepfilter-post-filter-beta", type=float, default=0.02,
                        help="DeepFilter post-filter beta")
    parser.add_argument("--deepfilter-atten-lim-db", type=float, default=30.0,
                        help="DeepFilter attenuation limit in dB")
    parser.add_argument("--deepfilter-compensate-delay", action=argparse.BooleanOptionalAction,
                        default=True, help="Compensate DeepFilter delay")
    parser.add_argument("--dereverb", action="store_true",
                        help="Enable dereverb placeholder step")
    parser.add_argument("--vocal-separation", action="store_true",
                        help="Enable Demucs vocal separation")
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument("--normalize", default=DEFAULT_NORMALIZE_MODE,
                        help="Normalize mode: none, peak, loudness, or chained values like loudness|peak")
    parser.add_argument("--peak-target-db", type=float, default=-1.0)
    parser.add_argument("--loudness-target-lufs", type=float, default=-16.0)
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    options = build_default_preprocess_options(True, args.input)
    options = apply_cli_overrides(options, args)
    output_path = preprocess_audio(args.input, options)
    print(output_path)


if __name__ == "__main__":
    main()
