import sys
import types
from importlib.machinery import ModuleSpec

if "torchcodec" not in sys.modules:
    try:
        _m = types.ModuleType("torchcodec")
        _m.__spec__ = ModuleSpec("torchcodec", None, origin="built-in")
        _m.__version__ = "0.2.0"
        _d = types.ModuleType("torchcodec.decoders")
        class MockDecoder: pass
        _d.AudioDecoder = MockDecoder
        _d.VideoDecoder = MockDecoder
        _m.decoders = _d
        sys.modules["torchcodec"] = _m
        sys.modules["torchcodec.decoders"] = _d
    except Exception:
        pass

import gc
import json
import logging
import os
import uuid
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import BitsAndBytesConfig

import folder_paths

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

folder_paths.add_model_folder_path("HeartMuLa", os.path.join(folder_paths.models_dir, "HeartMuLa"))
folder_paths.add_model_folder_path("HeartMuLa", os.path.join(current_dir, "util", "heartlib", "ckpt"))

def get_model_base_dir():
    paths = folder_paths.get_folder_paths("HeartMuLa")
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

MODEL_BASE_DIR = get_model_base_dir()

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _get_dtype(device: torch.device):
    if device.type == "mps":
        return torch.float32  
    return torch.bfloat16

class HeartMuLaModelManager:
    _instance = None
    _gen_pipes = {}
    _transcribe_pipe = None
    _device = _get_device()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HeartMuLaModelManager, cls).__new__(cls)
        return cls._instance

    def get_gen_pipeline(self, version="3B", codec_version="oss", quantize_4bit=False):
        if codec_version is None or str(codec_version).lower() == "none" or codec_version == "":
            codec_version = "oss"

        key = (version, codec_version, quantize_4bit)
        if key not in self._gen_pipes:
            from heartlib import HeartMuLaGenPipeline

            model_dtype = _get_dtype(self._device)

            bnb_config = None
            if quantize_4bit:
                if self._device.type != "cuda":
                    print(f"HeartMuLa: 4-bit quantization requires CUDA.")
                else:
                    quant_type = "nf4"
                    try:
                        major, _ = torch.cuda.get_device_capability()
                        if major >= 10:
                            quant_type = "fp4"
                    except: pass

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )

            self._gen_pipes[key] = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                torch_dtype=model_dtype,
                version=version,
                codec_version=codec_version,
                lazy_load=True,
                bnb_config=bnb_config
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return self._gen_pipes[key]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            from heartlib import HeartTranscriptorPipeline
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR, device=self._device, dtype=torch.float16,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return self._transcribe_pipe

class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                "version": (["3B", "7B", "RL-oss-3B-20260123"], {"default": "3B"}),
                "codec_version": (["oss", "oss-20260123"], {"default": "oss"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_audio_length_seconds": ("INT", {"default": 240, "min": 10, "max": 600, "step": 1}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "offload_mode": (["auto", "aggressive"], {"default": "auto"}),
                "quantize_4bit": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, codec_version, seed, max_audio_length_seconds, topk, temperature, cfg_scale, keep_model_loaded, offload_mode="auto", quantize_4bit=False):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        max_audio_length_ms = int(max_audio_length_seconds * 1000)
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version, codec_version=codec_version, quantize_4bit=quantize_4bit)

        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        try:
            with torch.inference_mode():
                pipe({"lyrics": lyrics, "tags": tags}, max_audio_length_ms=max_audio_length_ms, save_path=out_path, topk=topk, temperature=temperature, cfg_scale=cfg_scale, keep_model_loaded=keep_model_loaded, offload_mode=offload_mode)
        except Exception as e:
            print(f"Generation failed: {e}")
            raise e
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        try:
            waveform, sample_rate = torchaudio.load(out_path)
            waveform = waveform.float()
            if waveform.ndim == 2: waveform = waveform.unsqueeze(0)
        except Exception:
            waveform_np, sample_rate = sf.read(out_path)
            if waveform_np.ndim == 1: waveform_np = waveform_np[np.newaxis, :]
            else: waveform_np = waveform_np.T
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 2: waveform = waveform.unsqueeze(0)

        audio_output = {"waveform": waveform, "sample_rate": sample_rate}
        return (audio_output, out_path)


# -----------------------------------------------------------------------------
# Timestamp Formatting Utilities for Karaoke/Subtitle Output
# -----------------------------------------------------------------------------
# These functions convert Whisper's word-level timestamp output into standard
# formats used by karaoke players, video editors, and other applications.
# Whisper extracts word timing from cross-attention weights during decoding.
# -----------------------------------------------------------------------------

def _format_srt_time(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    if seconds is None:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def _format_lrc_time(seconds):
    """Convert seconds to LRC timestamp format: [mm:ss.xx]"""
    if seconds is None:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _format_timestamps_srt(chunks):
    """
    Format word timestamps as SubRip (SRT) subtitle format.
    
    SRT is widely supported by video players and editors. Each word gets
    its own subtitle entry with precise timing for karaoke-style display.
    
    Output format:
        1
        00:00:00,000 --> 00:00:00,520
        Hello
        
        2
        00:00:00,520 --> 00:00:01,040
        world
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk.get("timestamp", (0.0, 0.0))
        # Handle None timestamps (can occur at audio boundaries)
        if start is None:
            start = 0.0
        if end is None:
            end = start + 0.5  # Default 500ms duration
        text = chunk.get("text", "").strip()
        if text:
            lines.append(str(i))
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(text)
            lines.append("")
    return "\n".join(lines)


def _format_timestamps_lrc(chunks):
    """
    Format word timestamps as Enhanced LRC format for karaoke applications.
    
    LRC (LyRiCs) format is the standard for karaoke players. This uses
    word-level timing where each word is prefixed with its start time.
    
    Output format:
        [00:00.00]Hello [00:00.52]world [00:01.04]how [00:01.48]are [00:02.00]you
    
    Karaoke players highlight each word as its timestamp is reached.
    """
    words = []
    for chunk in chunks:
        start, _ = chunk.get("timestamp", (0.0, 0.0))
        if start is None:
            start = 0.0
        text = chunk.get("text", "").strip()
        if text:
            words.append(f"[{_format_lrc_time(start)}]{text}")
    return " ".join(words)


def _format_timestamps_json(chunks):
    """
    Format word timestamps as JSON for programmatic use.
    
    Returns a JSON array with word, start time, and end time for each word.
    Useful for building custom karaoke UIs or further processing.
    
    Output format:
        [
          {"word": "Hello", "start": 0.0, "end": 0.52},
          {"word": "world", "start": 0.52, "end": 1.04}
        ]
    """
    result = []
    for chunk in chunks:
        start, end = chunk.get("timestamp", (0.0, 0.0))
        text = chunk.get("text", "").strip()
        if text:
            result.append({
                "word": text,
                "start": start if start is not None else 0.0,
                "end": end if end is not None else (start or 0.0) + 0.5
            })
    return json.dumps(result, indent=2, ensure_ascii=False)


class HeartMuLa_Transcribe:
    """
    Lyrics transcription node using HeartTranscriptor (Whisper-based).
    
    Supports word-level timestamps for karaoke applications. Whisper extracts
    timing information from cross-attention weights during decoding, providing
    ~50-150ms accuracy which is sufficient for karaoke highlighting.
    
    Output formats:
    - plain: Just the transcribed text (default)
    - srt: SubRip subtitle format for video players
    - lrc: LRC format for karaoke players (word-level timing)
    - json: Structured JSON for programmatic use
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "temperature_tuple": ("STRING", {"default": "0.0,0.1,0.2,0.4"}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                # Word-level timestamps for karaoke - uses Whisper's cross-attention
                "return_timestamps": (["none", "word"], {"default": "none"}),
                # Output format selection
                "output_format": (["plain", "srt", "lrc", "json"], {"default": "plain"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, audio_input, temperature_tuple, no_speech_threshold, logprob_threshold, 
                   return_timestamps="none", output_format="plain"):
        try:
            torchaudio.set_audio_backend("soundfile")
        except: pass

        if isinstance(audio_input, dict):
            waveform, sr = audio_input["waveform"], audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
            if isinstance(waveform, np.ndarray): waveform = torch.from_numpy(waveform)

        if waveform.ndim == 3: waveform = waveform.squeeze(0)
        elif waveform.ndim == 1: waveform = waveform.unsqueeze(0)

        waveform = waveform.to(torch.float32).cpu()
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"hm_trans_{uuid.uuid4().hex}.wav")

        wav_np = waveform.numpy()
        if wav_np.ndim == 2: wav_np = wav_np.T
        sf.write(temp_path, wav_np, sr)

        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except: temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        # Determine timestamp mode for Whisper pipeline
        # "word" returns word-level timestamps from cross-attention weights
        timestamps_param = "word" if return_timestamps == "word" else False

        try:
            with torch.inference_mode():
                result = pipe(
                    temp_path, 
                    temperature=temp_tuple, 
                    no_speech_threshold=no_speech_threshold, 
                    logprob_threshold=logprob_threshold, 
                    task="transcribe",
                    return_timestamps=timestamps_param
                )
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        # Format output based on user selection
        if return_timestamps == "word" and isinstance(result, dict) and "chunks" in result:
            chunks = result["chunks"]
            
            if output_format == "srt":
                text = _format_timestamps_srt(chunks)
            elif output_format == "lrc":
                text = _format_timestamps_lrc(chunks)
            elif output_format == "json":
                text = _format_timestamps_json(chunks)
            else:  # plain - include timestamps inline
                text = result.get("text", str(result))
        else:
            # No timestamps requested or not available - return plain text
            text = result if isinstance(result, str) else result.get("text", str(result))
        
        return (text,)

NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate": HeartMuLa_Generate,
    "HeartMuLa_Transcribe": HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
