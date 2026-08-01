import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1]))

from minimax_music_cover import (  # noqa: E402
    MiniMax_MusicCover,
    _audio_bytes_from_response,
    _request_music_cover,
)


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


class MiniMaxMusicCoverTests(unittest.TestCase):
    def test_request_uses_bearer_auth_and_cover_fields(self):
        response = _Response(
            json.dumps(
                {
                    "base_resp": {"status_code": 0},
                    "data": {"status": 2, "audio": "00ff"},
                }
            ).encode()
        )
        payload = {
            "model": "music-cover",
            "audio_base64": "ZmFrZQ==",
            "cover_feature_id": "feature-id",
        }

        with patch(
            "minimax_music_cover.urllib.request.urlopen", return_value=response
        ) as open_url:
            self.assertEqual(
                _request_music_cover(
                    "https://api.minimax.io/v1/music_generation",
                    "test-key",
                    payload,
                ),
                "00ff",
            )

        request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), payload)

    def test_url_audio_is_downloaded(self):
        with patch(
            "minimax_music_cover.urllib.request.urlopen",
            return_value=_Response(b"audio-bytes"),
        ) as open_url:
            self.assertEqual(
                _audio_bytes_from_response("https://cdn.example/audio.mp3"),
                b"audio-bytes",
            )
        self.assertEqual(open_url.call_args.args[0], "https://cdn.example/audio.mp3")

    def test_hex_audio_is_decoded(self):
        self.assertEqual(_audio_bytes_from_response("00 ff 10"), b"\x00\xff\x10")

    def test_node_exposes_cover_models_and_sources(self):
        required = MiniMax_MusicCover.INPUT_TYPES()["required"]
        self.assertEqual(required["model"][0], ["music-cover", "music-cover-free"])
        self.assertIn("audio_url", required)
        self.assertIn("audio_base64", required)
        self.assertIn("cover_feature_id", required)

    def test_node_requires_exactly_one_audio_source(self):
        node = MiniMax_MusicCover()
        common_args = (
            "test-key",
            "music-cover",
            "global_en",
        )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            node.generate(*common_args, "", "", "feature-id", "url")

        with self.assertRaisesRegex(ValueError, "exactly one"):
            node.generate(
                *common_args,
                "https://cdn.example/input.mp3",
                "ZmFrZQ==",
                "feature-id",
                "url",
            )


if __name__ == "__main__":
    unittest.main()
