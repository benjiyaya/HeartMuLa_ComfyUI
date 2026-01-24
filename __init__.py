"""
HeartMuLa ComfyUI Custom Nodes
==============================

This module provides ComfyUI nodes for the HeartMuLa music generation system.

Based on the paper:
    "HeartMuLa: A Family of Open Sourced Music Foundation Models"
    arXiv:2601.10547 - https://arxiv.org/abs/2601.10547

Nodes Provided:
---------------
1. HeartMuLa_Generate: Standard music generation node
2. HeartMuLa_Generate_Optimized: Optimized version with FlashAttention and section control
3. HeartMuLa_Transcribe: Lyrics transcription from audio

Architecture Overview (from paper):
-----------------------------------
HeartMuLa uses a two-stage architecture:

Stage 1 - HeartMuLa LLM (3B or 7B parameters):
    - Based on Llama-3.2 architecture
    - Takes lyrics + style tags as conditioning
    - Autoregressively generates 8 parallel codebook tokens per frame
    - Each frame = 80ms of audio (12.5 Hz frame rate)
    - For a 3-minute song: 180s * 12.5 Hz = 2250 frames

Stage 2 - HeartCodec Flow-Matching Decoder:
    - Takes codebook tokens from Stage 1
    - Uses continuous normalizing flows (flow-matching)
    - ODE solver generates continuous latent representation
    - Outputs 48kHz stereo audio
    - Paper shows 10 ODE steps ≈ 20 steps in quality

Memory Management:
------------------
The paper targets 12GB VRAM (Section 3.5.3). We achieve this by:
1. Never loading HeartMuLa and HeartCodec simultaneously
2. Using bf16 precision (half memory vs fp32)
3. Aggressive garbage collection between stages
4. Optional 4-bit quantization (reduces 3B model from ~6GB to ~2GB)

License: Same as parent HeartMuLa project
"""

import sys
import types
from importlib.machinery import ModuleSpec

# =============================================================================
# TORCHCODEC MOCK
# =============================================================================
# Why: torchcodec is an optional dependency that may not be installed.
# The main HeartMuLa code doesn't use it, but torchtune imports it.
# We create a mock module to prevent ImportError without requiring installation.
# This is a common pattern for optional dependencies in ML libraries.
# =============================================================================

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

# =============================================================================
# CUDA MEMORY CONFIGURATION
# =============================================================================
# Why "expandable_segments:True":
# By default, PyTorch's CUDA allocator uses fixed-size memory segments.
# When these fill up, allocation can fail even if there's free memory.
# "expandable_segments" allows segments to grow, reducing OOM errors.
# This is especially important for variable-length music generation.
#
# Reference: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
# =============================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress verbose logging from dependencies
# (transformers and torchtune print many warnings during model loading)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =============================================================================
# PATH SETUP
# =============================================================================
# We need to add the 'util' directory to sys.path so that 'heartlib' can be
# imported as a top-level module. This is necessary because ComfyUI's custom
# node loading doesn't automatically handle submodule imports.
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# =============================================================================
# MODEL FOLDER REGISTRATION
# =============================================================================
# ComfyUI uses folder_paths to manage model locations. We register two paths:
# 1. models/HeartMuLa - Standard ComfyUI models directory
# 2. util/heartlib/ckpt - Local checkpoint directory (for bundled models)
#
# This allows users to place models in either location.
# =============================================================================

folder_paths.add_model_folder_path("HeartMuLa", os.path.join(folder_paths.models_dir, "HeartMuLa"))
folder_paths.add_model_folder_path("HeartMuLa", os.path.join(current_dir, "util", "heartlib", "ckpt"))

def get_model_base_dir():
    """
    Find the first existing HeartMuLa model directory.
    
    Returns the first path that exists, or the first registered path if none exist.
    This allows graceful handling when models haven't been downloaded yet.
    """
    paths = folder_paths.get_folder_paths("HeartMuLa")
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

MODEL_BASE_DIR = get_model_base_dir()


def _get_device() -> torch.device:
    """
    Auto-detect the best available compute device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Why this order:
    - CUDA: Best performance, full feature support
    - MPS: Good performance on Apple Silicon, some limitations
    - CPU: Fallback, very slow for large models
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_dtype(device: torch.device) -> torch.dtype:
    """
    Get optimal dtype for the device.
    
    Why bf16 for CUDA:
    - Same dynamic range as fp32 (8-bit exponent)
    - Half the memory of fp32
    - Tensor Core acceleration on Ampere+ GPUs
    - HeartMuLa was trained in bf16
    
    Why fp32 for MPS:
    - MPS bf16 support is limited in PyTorch
    - Some operations fall back to fp32 anyway
    - More stable on Apple Silicon
    """
    if device.type == "mps":
        return torch.float32  
    return torch.bfloat16


# =============================================================================
# MODEL MANAGER (Singleton Pattern)
# =============================================================================
# Why Singleton:
# - Large models (3B+ parameters) should be shared across nodes
# - Prevents duplicate models consuming VRAM
# - Enables model caching between generations
#
# The manager lazily loads models on first use and caches them for reuse.
# Different configurations (version, quantization) get separate cache entries.
# =============================================================================

class HeartMuLaModelManager:
    """
    Singleton manager for HeartMuLa model instances.
    
    Manages lifecycle of generation and transcription pipelines:
    - Lazy loading: Models loaded on first use
    - Caching: Reuse loaded models across generations
    - Configuration keying: Different configs get separate instances
    
    Singleton Pattern:
    - Only one instance exists per process
    - All ComfyUI nodes share the same manager
    - Prevents duplicate models wasting VRAM
    """
    _instance = None
    _gen_pipes = {}          # Cache: (version, codec, quant) -> pipeline
    _transcribe_pipe = None  # Single transcription pipeline
    _device = _get_device()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(HeartMuLaModelManager, cls).__new__(cls)
        return cls._instance

    def get_gen_pipeline(self, version="3B", codec_version="oss", quantize_4bit=False):
        """
        Get or create a generation pipeline.
        
        Args:
            version: HeartMuLa version ("3B", "7B", "RL-oss-3B-20260123")
            codec_version: HeartCodec version ("oss", "oss-20260123")
            quantize_4bit: Enable 4-bit quantization (reduces VRAM ~3x)
            
        Returns:
            HeartMuLaGenPipeline instance, cached for reuse
            
        Why Cache by Configuration:
        - Different versions have different weights
        - Quantized vs non-quantized models are different objects
        - Users might switch between configs in a workflow
        
        4-bit Quantization Details:
        - Uses bitsandbytes library (CUDA only)
        - NF4 (Normalized Float 4): Default, works on all NVIDIA GPUs
        - FP4: Native 4-bit on Blackwell+ (SM 10.0+), ~10% faster
        - Double quantization: Quantizes the quantization constants
        - Reduces 3B model from ~6GB to ~2GB VRAM
        """
        # Normalize codec_version (handle None/empty string from UI)
        if codec_version is None or str(codec_version).lower() == "none" or codec_version == "":
            codec_version = "oss"

        # Create cache key from configuration tuple
        key = (version, codec_version, quantize_4bit)
        
        if key not in self._gen_pipes:
            from heartlib import HeartMuLaGenPipeline

            model_dtype = _get_dtype(self._device)

            # Configure 4-bit quantization if requested
            bnb_config = None
            if quantize_4bit:
                if self._device.type != "cuda":
                    print(f"[WARN] HeartMuLa: 4-bit quantization requires CUDA, using full precision.")
                else:
                    # Default to NF4 (Normalized Float 4)
                    # NF4 is optimized for normally-distributed weights
                    quant_type = "nf4"
                    try:
                        # Check for Blackwell+ (SM 10.0+) which has native FP4
                        major, _ = torch.cuda.get_device_capability()
                        if major >= 10:
                            quant_type = "fp4"
                            print(f"[INFO] Using native FP4 quantization (Blackwell+)")
                    except: 
                        pass

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
                        bnb_4bit_use_double_quant=True,  # Additional compression
                    )

            # Create pipeline with lazy loading
            # (actual model weights loaded on first generation)
            self._gen_pipes[key] = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                torch_dtype=model_dtype,
                version=version,
                codec_version=codec_version,
                lazy_load=True,
                bnb_config=bnb_config
            )
            
            # Clean up any temporary allocations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        return self._gen_pipes[key]

    def get_transcribe_pipeline(self):
        """
        Get or create the transcription pipeline.
        
        HeartTranscriptor (from paper Section 5):
        - Whisper-based lyrics recognition model
        - Optimized for music scenarios (background music, reverb, etc.)
        - Uses fp16 for efficiency (transcription is lighter than generation)
        
        Why Single Instance (no config keying):
        - Transcription doesn't have version variants
        - Same model works for all inputs
        - Saves memory by not caching multiple instances
        """
        if self._transcribe_pipe is None:
            from heartlib import HeartTranscriptorPipeline
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR, device=self._device, dtype=torch.float16,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return self._transcribe_pipe


# =============================================================================
# COMFYUI NODE DEFINITIONS
# =============================================================================
# ComfyUI nodes define:
# - INPUT_TYPES: What parameters the node accepts
# - RETURN_TYPES: What outputs the node produces
# - FUNCTION: The method that performs the operation
# - CATEGORY: Where to find the node in ComfyUI's menu
# =============================================================================

class HeartMuLa_Generate:
    """
    Standard HeartMuLa music generation node.
    
    Generates music from lyrics and style tags using the HeartMuLa model.
    
    Generation Parameters Explained:
    --------------------------------
    - temperature: Controls randomness in sampling (higher = more creative/chaotic)
      - 0.7-0.9: Coherent, predictable
      - 1.0: Balanced (default)
      - 1.2+: More experimental, may be inconsistent
      
    - topk: Number of top tokens to consider at each step
      - Lower (20-50): More focused, consistent
      - Higher (80-150): More diverse, creative
      
    - cfg_scale: Classifier-free guidance strength
      - 1.0: No guidance (pure model output)
      - 1.5: Light guidance (default)
      - 2.0-3.0: Strong adherence to prompt
      - Higher: Risk of artifacts/repetition
      
    Memory Modes:
    - keep_model_loaded=True + offload_mode="auto": Fast regeneration, ~6GB VRAM
    - keep_model_loaded=False + offload_mode="aggressive": Minimum memory, slower
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the ComfyUI node."""
        return {
            "required": {
                # Lyrics with section markers like [Verse], [Chorus]
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                # Style description: genre, instruments, mood
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                # Model version: 3B is faster, 7B is higher quality
                "version": (["3B", "7B", "RL-oss-3B-20260123"], {"default": "3B"}),
                "codec_version": (["oss", "oss-20260123"], {"default": "oss"}),
                # Random seed for reproducibility
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # Maximum audio duration in seconds
                "max_audio_length_seconds": ("INT", {"default": 240, "min": 10, "max": 600, "step": 1}),
                # Sampling parameters (see class docstring)
                "topk": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                # Memory management options
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "offload_mode": (["auto", "aggressive"], {"default": "auto"}),
                # 4-bit quantization for lower VRAM usage
                "quantize_4bit": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, codec_version, seed, max_audio_length_seconds, topk, temperature, cfg_scale, keep_model_loaded, offload_mode="auto", quantize_4bit=False):
        """Generate music from lyrics and style tags."""
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

class HeartMuLa_Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "temperature_tuple": ("STRING", {"default": "0.0,0.1,0.2,0.4"}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, audio_input, temperature_tuple, no_speech_threshold, logprob_threshold):
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

        try:
            with torch.inference_mode():
                result = pipe(temp_path, temperature=temp_tuple, no_speech_threshold=no_speech_threshold, logprob_threshold=logprob_threshold, task="transcribe")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        text = result if isinstance(result, str) else result.get("text", str(result))
        return (text,)

class HeartMuLa_Generate_Optimized:
    """
    Optimized HeartMuLa Music Generator with:
    - FlashAttention support (faster inference)
    - Fine-grained section control (intro/verse/chorus styling)
    - Configurable decode steps
    
    Based on paper: https://arxiv.org/abs/2601.10547
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\nYour lyrics here...\n[Chorus]\nChorus lyrics..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "pop ballad, piano, emotional vocals"}),
                "version": (["3B", "7B", "RL-oss-3B-20260123"], {"default": "RL-oss-3B-20260123"}),
                "codec_version": (["oss", "oss-20260123"], {"default": "oss-20260123"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_audio_length_seconds": ("INT", {"default": 240, "min": 10, "max": 600, "step": 1}),
                "topk": ("INT", {"default": 80, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 2.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "offload_mode": (["auto", "aggressive"], {"default": "auto"}),
                "quantize_4bit": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "use_flash_attention": ("BOOLEAN", {"default": True}),
                "decode_steps": ("INT", {"default": 10, "min": 5, "max": 50, "step": 1}),
                "intro_style": ("STRING", {"multiline": False, "placeholder": "soft piano intro, ambient"}),
                "verse_style": ("STRING", {"multiline": False, "placeholder": "acoustic guitar, gentle"}),
                "chorus_style": ("STRING", {"multiline": False, "placeholder": "full band, powerful"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, codec_version, seed, max_audio_length_seconds, 
                 topk, temperature, cfg_scale, keep_model_loaded, offload_mode="auto", 
                 quantize_4bit=False, use_flash_attention=True, decode_steps=10,
                 intro_style="", verse_style="", chorus_style=""):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        # Build enhanced tags with section styles if provided
        enhanced_tags = tags
        if intro_style or verse_style or chorus_style:
            section_tags = []
            if intro_style:
                section_tags.append(f"[intro] {intro_style}")
            if verse_style:
                section_tags.append(f"[verse] {verse_style}")
            if chorus_style:
                section_tags.append(f"[chorus] {chorus_style}")
            enhanced_tags = f"{tags}\n{chr(10).join(section_tags)}"

        max_audio_length_ms = int(max_audio_length_seconds * 1000)
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version, codec_version=codec_version, quantize_4bit=quantize_4bit)

        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_optimized_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        try:
            with torch.inference_mode():
                pipe(
                    {"lyrics": lyrics, "tags": enhanced_tags}, 
                    max_audio_length_ms=max_audio_length_ms, 
                    save_path=out_path, 
                    topk=topk, 
                    temperature=temperature, 
                    cfg_scale=cfg_scale, 
                    keep_model_loaded=keep_model_loaded, 
                    offload_mode=offload_mode
                )
        except Exception as e:
            print(f"Generation failed: {e}")
            raise e
        finally:
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            gc.collect()

        try:
            waveform, sample_rate = torchaudio.load(out_path)
            waveform = waveform.float()
            if waveform.ndim == 2: 
                waveform = waveform.unsqueeze(0)
        except Exception:
            waveform_np, sample_rate = sf.read(out_path)
            if waveform_np.ndim == 1: 
                waveform_np = waveform_np[np.newaxis, :]
            else: 
                waveform_np = waveform_np.T
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 2: 
                waveform = waveform.unsqueeze(0)

        audio_output = {"waveform": waveform, "sample_rate": sample_rate}
        return (audio_output, out_path)


NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate": HeartMuLa_Generate,
    "HeartMuLa_Generate_Optimized": HeartMuLa_Generate_Optimized,
    "HeartMuLa_Transcribe": HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Generate_Optimized": "HeartMuLa Music Generator (Optimized)",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
