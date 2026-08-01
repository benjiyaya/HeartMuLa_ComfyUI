import json
import os
import urllib.error
import urllib.request
import uuid


MINIMAX_MUSIC_COVER_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}
MINIMAX_MUSIC_COVER_MODELS = ("music-cover", "music-cover-free")
MINIMAX_MUSIC_COVER_OUTPUT_FORMATS = ("url", "hex")


def _request_music_cover(endpoint, api_key, payload):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"MiniMax music cover request failed with HTTP {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("MiniMax music cover request failed.") from exc

    try:
        response_json = json.loads(response_body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MiniMax music cover returned invalid JSON.") from exc

    base_response = response_json.get("base_resp") or {}
    if base_response.get("status_code") != 0:
        message = base_response.get("status_msg") or "unknown API error"
        raise RuntimeError(f"MiniMax music cover request failed: {message}")

    data = response_json.get("data") or {}
    if data.get("status") not in (None, 2, "2"):
        raise RuntimeError("MiniMax music cover did not return completed audio.")

    audio = data.get("audio")
    if not isinstance(audio, str) or not audio:
        raise RuntimeError("MiniMax music cover response did not include audio.")
    return audio


def _audio_bytes_from_response(audio):
    if not isinstance(audio, str) or not audio:
        raise ValueError("MiniMax music cover returned an empty audio value.")

    if audio.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(audio, timeout=300) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError("MiniMax music cover audio download failed.") from exc

    try:
        return bytes.fromhex(audio)
    except ValueError as exc:
        raise ValueError("MiniMax music cover audio was neither a URL nor hex.") from exc


class MiniMax_MusicCover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    list(MINIMAX_MUSIC_COVER_MODELS),
                    {"default": "music-cover"},
                ),
                "region": (
                    list(MINIMAX_MUSIC_COVER_ENDPOINTS),
                    {"default": "global_en"},
                ),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
                "audio_base64": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "cover_feature_id": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "output_format": (
                    list(MINIMAX_MUSIC_COVER_OUTPUT_FORMATS),
                    {"default": "url"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(
        self,
        api_key,
        model,
        region,
        audio_url,
        audio_base64,
        cover_feature_id,
        output_format,
    ):
        api_key = (api_key or "").strip()
        audio_url = (audio_url or "").strip()
        audio_base64 = "".join((audio_base64 or "").split())
        cover_feature_id = (cover_feature_id or "").strip()

        if not api_key:
            raise ValueError("MiniMax API key is required.")
        if model not in MINIMAX_MUSIC_COVER_MODELS:
            raise ValueError(f"Unsupported MiniMax music cover model: {model}")
        if region not in MINIMAX_MUSIC_COVER_ENDPOINTS:
            raise ValueError(f"Unsupported MiniMax region: {region}")
        if output_format not in MINIMAX_MUSIC_COVER_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported MiniMax output format: {output_format}")
        if not cover_feature_id:
            raise ValueError("MiniMax cover feature ID is required.")
        if bool(audio_url) == bool(audio_base64):
            raise ValueError("Provide exactly one of audio_url or audio_base64.")

        payload = {
            "model": model,
            "cover_feature_id": cover_feature_id,
            "output_format": output_format,
            "audio_setting": {"format": "mp3"},
        }
        if audio_url:
            payload["audio_url"] = audio_url
        else:
            payload["audio_base64"] = audio_base64

        audio = _request_music_cover(
            MINIMAX_MUSIC_COVER_ENDPOINTS[region], api_key, payload
        )
        audio_bytes = _audio_bytes_from_response(audio)

        import folder_paths
        import torch
        import torchaudio

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"minimax_music_cover_{uuid.uuid4().hex}.mp3"
        )
        with open(output_path, "wb") as output_file:
            output_file.write(audio_bytes)

        waveform, sample_rate = torchaudio.load(output_path)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sample_rate}, output_path)
