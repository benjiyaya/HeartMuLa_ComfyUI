import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import gc
import folder_paths

# ----------------------------
# Add Local HeartLib to Path
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# ----------------------------
# Path Configuration
# ----------------------------
MODEL_BASE_DIR = os.path.join(folder_paths.models_dir, "HeartMuLa")

# ----------------------------
# Global Model Manager
# ----------------------------
class HeartMuLaModelManager:
    _instance = None
    _gen_pipes = {}
    _transcribe_pipe = None 
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HeartMuLaModelManager, cls).__new__(cls)
        return cls._instance

    def get_gen_pipeline(self, version="3B"):
        if version not in self._gen_pipes or self._gen_pipes[version] is None:
            print(f"[HeartMuLa] Loading Generation Pipeline (Version: {version}) on {self._device}...")
            from heartlib import HeartMuLaGenPipeline
            
            self._gen_pipes[version] = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.bfloat16,
                version=version,
            )
            print(f"[HeartMuLa] Generation Pipeline ({version}) Ready.")
            
        return self._gen_pipes[version]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            print(f"[HeartMuLa] Loading Transcription Pipeline on {self._device}...")
            from heartlib import HeartTranscriptorPipeline
            
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.float16,
            )
            print("[HeartMuLa] Transcription Pipeline Ready.")
            
        return self._transcribe_pipe

    def unload_gen_pipeline(self, version):
        """Unload specific generation pipeline to free VRAM"""
        if version in self._gen_pipes and self._gen_pipes[version] is not None:
            print(f"[HeartMuLa] Unloading Generation Pipeline ({version}) to free VRAM...")
            del self._gen_pipes[version]
            self._gen_pipes[version] = None
            torch.cuda.synchronize() # Wait for GPU ops to finish
            torch.cuda.empty_cache()
            gc.collect()

    def unload_transcribe_pipeline(self):
        """Unload transcription pipeline to free VRAM"""
        if self._transcribe_pipe is not None:
            print("[HeartMuLa] Unloading Transcription Pipeline to free VRAM...")
            del self._transcribe_pipe
            self._transcribe_pipe = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

# ----------------------------
# Node: Music Generator
# ----------------------------
class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                "version": (["3B", "7B"], {"default": "3B"}),
                # Optimized Default: 60s is safer for 16GB VRAM than 240s
                "max_audio_length_ms": ("INT", {"default": 60000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_model_after_run": ("BOOLEAN", {"default": True}), # MEMORY OPT
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, max_audio_length_ms, topk, temperature, cfg_scale, seed, control_seed, unload_model_after_run):
        # --- SEED HANDLING ---
        if seed > 0:
            torch.manual_seed(seed)
        elif control_seed > 0:
            torch.manual_seed(control_seed)
        else:
            print("[HeartMuLa] Random Seed used.")
        
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        # --- GENERATION ---
        with torch.no_grad():
            pipe(
                {"lyrics": lyrics, "tags": tags},
                max_audio_length_ms=max_audio_length_ms,
                save_path=out_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
        
        # --- MEMORY CLEANUP (Optimized for 16GB) ---
        torch.cuda.empty_cache()
        gc.collect()

        # --- UNLOAD LOGIC ---
        if unload_model_after_run:
            manager.unload_gen_pipeline(version)

        # Load result back for ComfyUI
        waveform, sample_rate = torchaudio.load(out_path)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        
        waveform = waveform.float()
            
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        audio_output = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        return (audio_output, out_path)

# ----------------------------
# Node: Lyrics Transcriber
# ----------------------------
class HeartMuLa_Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "temperature_tuple": ("STRING", {"default": "0.0,0.1,0.2,0.4"}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "unload_model_after_run": ("BOOLEAN", {"default": True}), # MEMORY OPT
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, audio_input, temperature_tuple, no_speech_threshold, logprob_threshold, unload_model_after_run):
        # --- PRE-PROCESSING ---
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
            if isinstance(waveform, np.ndarray):
                 waveform = torch.from_numpy(waveform)
        
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        temp_filename = f"heartmula_transcribe_in_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(output_dir, temp_filename)

        torchaudio.save(temp_path, waveform, sr)

        # --- INFERENCE ---
        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except:
            temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        with torch.no_grad():
            result = pipe(
                temp_path,
                temperature=temp_tuple,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=1.8,
                max_new_tokens=256,
                num_beams=2,
                task="transcribe",
                condition_on_prev_tokens=False
            )

        # --- MEMORY CLEANUP ---
        torch.cuda.empty_cache()
        gc.collect()

        # --- UNLOAD LOGIC ---
        if unload_model_after_run:
            manager.unload_transcribe_pipeline()

        # --- POST-PROCESSING ---
        if os.path.exists(temp_path):
            os.remove(temp_path)

        text = result if isinstance(result, str) else result.get("text", str(result))
        return (text,)

# ----------------------------
# Node Mappings
# ----------------------------
NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate": HeartMuLa_Generate,
    "HeartMuLa_Transcribe": HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
