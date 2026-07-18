import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import preprocess_audio


def _write_wav_file(path, sample_rate, channels, sample_width=2, frame_count=32):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frame_count * channels * sample_width)


class PreprocessAudioTests(unittest.TestCase):
    def test_default_normalize_mode_is_loudness_then_peak(self):
        self.assertEqual(preprocess_audio.DEFAULT_NORMALIZE_MODE, "loudness|peak")

    def test_parse_normalize_mode_supports_chained_steps(self):
        self.assertEqual(
            preprocess_audio._parse_normalize_mode("loudness|peak"),
            ["loudness", "peak"],
        )

    def test_parse_normalize_mode_preserves_custom_order(self):
        self.assertEqual(
            preprocess_audio._parse_normalize_mode("peak|loudness"),
            ["peak", "loudness"],
        )

    def test_parse_normalize_mode_deduplicates_repeated_steps(self):
        self.assertEqual(preprocess_audio._parse_normalize_mode("peak|peak"), ["peak"])

    def test_parse_normalize_mode_supports_none(self):
        self.assertEqual(preprocess_audio._parse_normalize_mode("none"), [])

    def test_parse_normalize_mode_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            preprocess_audio._parse_normalize_mode("compressor")

    def test_parse_normalize_mode_rejects_none_mixed_with_other_steps(self):
        with self.assertRaises(ValueError):
            preprocess_audio._parse_normalize_mode("none|peak")

    def test_apply_deepfilternet2_uses_cli_command(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            input_path.write_bytes(b"input")
            output_path = temp_path / "output.wav"
            produced_path = output_path.parent / "deepfilternet2" / input_path.name
            options = preprocess_audio.PreprocessOptions(enabled=True)

            def fake_run_command(command, step_name):
                self.assertEqual(step_name, "DeepFilterNet2 denoise")
                self.assertEqual(
                    command,
                    [
                        options.deepfilter_bin,
                        "--pf",
                        "--pf-beta",
                        str(options.deepfilter_post_filter_beta),
                        "-a",
                        str(options.deepfilter_atten_lim_db),
                        "-D",
                        str(input_path),
                        "--output-dir",
                        str(output_path.parent / "deepfilternet2"),
                    ],
                )
                produced_path.parent.mkdir(parents=True, exist_ok=True)
                produced_path.write_bytes(b"denoised")

            with patch("preprocess_audio._run_command", side_effect=fake_run_command):
                result = preprocess_audio._apply_deepfilternet2(input_path, output_path, options)

            self.assertEqual(result, output_path)
            self.assertEqual(output_path.read_bytes(), b"denoised")

    def test_standardize_audio_skips_when_input_already_matches_target_wav(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "output.wav"
            _write_wav_file(input_path, sample_rate=16000, channels=1)
            options = preprocess_audio.PreprocessOptions(enabled=True)

            with patch("preprocess_audio._run_command") as run_command:
                result = preprocess_audio._standardize_audio(input_path, output_path, options)

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), input_path.read_bytes())
            run_command.assert_not_called()

    def test_copy_wav_file_copies_input_to_output(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "copied.wav"
            _write_wav_file(input_path, sample_rate=16000, channels=1)

            result = preprocess_audio._copy_wav_file(input_path, output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), input_path.read_bytes())

    def test_standardize_audio_only_changes_sample_rate_when_channels_already_match(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "resampled.wav"
            _write_wav_file(input_path, sample_rate=44100, channels=1)
            options = preprocess_audio.PreprocessOptions(enabled=True)

            with patch("preprocess_audio._run_command") as run_command:
                result = preprocess_audio._standardize_audio(input_path, output_path, options)

            self.assertEqual(result, output_path)
            run_command.assert_called_once_with(
                [
                    options.ffmpeg_bin,
                    "-y",
                    "-i",
                    str(input_path),
                    "-ar",
                    "16000",
                    str(output_path),
                ],
                "Audio resampling and channel conversion",
            )

    def test_standardize_audio_only_changes_channels_when_sample_rate_already_matches(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "mono.wav"
            _write_wav_file(input_path, sample_rate=16000, channels=2)
            options = preprocess_audio.PreprocessOptions(enabled=True)

            with patch("preprocess_audio._run_command") as run_command:
                result = preprocess_audio._standardize_audio(input_path, output_path, options)

            self.assertEqual(result, output_path)
            run_command.assert_called_once_with(
                [
                    options.ffmpeg_bin,
                    "-y",
                    "-i",
                    str(input_path),
                    "-ac",
                    str(options.channels),
                    str(output_path),
                ],
                "Audio resampling and channel conversion",
            )

    def test_preprocess_audio_uses_48k_before_denoise_and_resamples_back(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            input_path.write_bytes(b"input")
            work_dir = temp_path / "work"
            options = preprocess_audio.PreprocessOptions(
                enabled=True,
                enable_denoise=True,
                normalize="none",
                work_dir=str(work_dir),
            )
            sample_rates = []

            def fake_standardize_audio(source_path, output_path, options_arg, sample_rate=None):
                effective_sample_rate = sample_rate if sample_rate is not None else options_arg.sample_rate
                sample_rates.append(effective_sample_rate)
                output_path.write_bytes(f"sr:{effective_sample_rate}".encode("ascii"))
                return output_path

            def fake_denoise(source_path, output_path, options_arg):
                output_path.write_bytes(source_path.read_bytes() + b"|denoised")
                return output_path

            with patch("preprocess_audio._standardize_audio", side_effect=fake_standardize_audio):
                with patch("preprocess_audio._apply_deepfilternet2", side_effect=fake_denoise):
                    result = preprocess_audio.preprocess_audio(input_path, options)

            self.assertEqual(sample_rates, [preprocess_audio.DEEPFILTERNET2_SAMPLE_RATE, options.sample_rate])
            self.assertEqual(Path(result).name, "final_preprocessed.wav")
            self.assertTrue(Path(result).exists())

    def test_preprocess_audio_normalizes_before_denoise(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            input_path.write_bytes(b"input")
            work_dir = temp_path / "work"
            options = preprocess_audio.PreprocessOptions(
                enabled=True,
                enable_denoise=True,
                normalize="peak",
                work_dir=str(work_dir),
            )
            calls = []

            def fake_standardize_audio(source_path, output_path, options_arg, sample_rate=None):
                calls.append(("standardize", sample_rate if sample_rate is not None else options_arg.sample_rate))
                output_path.write_bytes(b"standardized")
                return output_path

            def fake_normalize_peak(source_path, output_path, options_arg):
                calls.append("peak_normalize")
                output_path.write_bytes(source_path.read_bytes() + b"|normalized")
                return output_path

            def fake_denoise(source_path, output_path, options_arg):
                calls.append("denoise")
                output_path.write_bytes(source_path.read_bytes() + b"|denoised")
                return output_path

            with patch("preprocess_audio._standardize_audio", side_effect=fake_standardize_audio):
                with patch("preprocess_audio._normalize_peak", side_effect=fake_normalize_peak):
                    with patch("preprocess_audio._apply_deepfilternet2", side_effect=fake_denoise):
                        preprocess_audio.preprocess_audio(input_path, options)

            self.assertEqual(calls[:3], [
                ("standardize", preprocess_audio.DEEPFILTERNET2_SAMPLE_RATE),
                "peak_normalize",
                "denoise",
            ])

    def test_preprocess_audio_keeps_original_when_standardize_step_is_skipped(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            _write_wav_file(input_path, sample_rate=16000, channels=1)
            work_dir = temp_path / "work"
            options = preprocess_audio.PreprocessOptions(
                enabled=True,
                enable_denoise=False,
                normalize="none",
                work_dir=str(work_dir),
            )

            result = preprocess_audio.preprocess_audio(input_path, options)

            self.assertTrue(input_path.exists())
            self.assertEqual(Path(result).name, "final_preprocessed.wav")
            self.assertTrue(Path(result).exists())


if __name__ == "__main__":
    unittest.main()