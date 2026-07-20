import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "funasr" not in sys.modules:
    sys.modules["funasr"] = types.SimpleNamespace(AutoModel=object)


subtitle_module = importlib.import_module("generate_subtitle")


class GenerateSubtitleTests(unittest.TestCase):
    def test_parse_timestamp_ms_supports_srt_and_vtt(self):
        self.assertEqual(subtitle_module.parse_timestamp_ms("00:01:02,345"), 62345)
        self.assertEqual(subtitle_module.parse_timestamp_ms("00:01:02.345"), 62345)

    def test_parse_json_from_text_extracts_embedded_json(self):
        payload = subtitle_module.parse_json_from_text(
            "response prefix {\"segments\": [{\"id\": 1, \"translation\": \"好\"}]} trailing text"
        )
        self.assertEqual(payload["segments"][0]["translation"], "好")

    def test_split_segments_for_translation_respects_budget(self):
        segments = [
            {"text": "a" * 8},
            {"text": "b" * 8},
            {"text": "c" * 8},
        ]
        groups = subtitle_module.split_segments_for_translation(segments, max_tokens=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0]["text"], "a" * 8)
        self.assertEqual(groups[1][0]["text"], "c" * 8)

    def test_build_segments_from_result_prefers_sentence_info(self):
        result_item = {
            "sentence_info": [
                {"sentence": "<|zh|>こんにちは", "start": 100, "end": 200, "spk": 1},
                {"sentence": "えー", "start": 210, "end": 260, "spk": 1},
            ]
        }
        segments = subtitle_module.build_segments_from_result(result_item, filter_fillers=True)
        self.assertEqual(segments, [{"start": 100, "end": 200, "text": "こんにちは", "spk": 1}])

    def test_build_segments_from_result_falls_back_to_result_text(self):
        result_item = {
            "text": "テスト",
            "timestamp": [{"start": 0.1, "end": 0.5}],
        }
        segments = subtitle_module.build_segments_from_result(result_item, filter_fillers=False)
        self.assertEqual(segments, [{"start": 100, "end": 500, "text": "テスト", "spk": None}])

    def test_extract_vad_intervals_supports_nested_funasr_value(self):
        intervals = subtitle_module.extract_vad_intervals([
            {"value": [[[0, 1200], [2000, 3500]]]},
        ])
        self.assertEqual(intervals, [(0, 1200), (2000, 3500)])

    def test_group_vad_intervals_splits_at_600_seconds(self):
        groups = subtitle_module.group_vad_intervals(
            [(0, 200000), (250000, 590000), (610000, 700000)],
            max_duration_ms=600000,
        )
        self.assertEqual(groups, [(0, 590000), (610000, 700000)])

    def test_prepare_transcription_input_extracts_mp4_audio(self):
        with mock.patch.object(subtitle_module.os, "makedirs") as make_dirs:
            with mock.patch.object(subtitle_module, "extract_mp4_audio") as extract:
                prepared_path = subtitle_module.prepare_transcription_input(
                    "video.MP4", "video.audio_preprocess")
                self.assertEqual(prepared_path, "video.audio_preprocess\\00_extracted.wav")
                make_dirs.assert_called_once_with("video.audio_preprocess", exist_ok=True)
                extract.assert_called_once_with("video.MP4", prepared_path)

    def test_append_user_prompt_keeps_and_appends_instructions(self):
        prompt = subtitle_module.append_user_prompt("Input: {}", "保留专有名词")
        self.assertIn("Input: {}", prompt)
        self.assertIn("Additional instructions:\n保留专有名词", prompt)

    def test_deepseek_thinking_is_only_added_for_deepseek_model(self):
        self.assertEqual(
            subtitle_module.build_deepseek_thinking_overrides("deepseek-v4-pro", True),
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )
        self.assertIsNone(
            subtitle_module.build_deepseek_thinking_overrides("gpt-5", True))

    def test_japanese_filler_patterns_match_individual_short_fillers(self):
        for filler in ("あ", "えー。", "えっと", "うーん", "んっ", "はぁ", "ふふ"):
            with self.subTest(filler=filler):
                self.assertTrue(subtitle_module.is_filler_only(filler))


if __name__ == "__main__":
    unittest.main()
