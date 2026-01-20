import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import gc
import folder_paths
import logging
import warnings


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

MODEL_BASE_DIR = os.path.join(folder_paths.models_dir, "HeartMuLa")

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
        if version not in self._gen_pipes:
            print(f"[HeartMuLa] Initializing Pipeline (Version: {version}) with Stage-Loading support...")
            from heartlib import HeartMuLaGenPipeline
            

            dtype = torch.bfloat16
            

            pipe = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=dtype,
                version=version,
                lazy_load=True
            )

            torch.cuda.empty_cache()
            gc.collect()

            self._gen_pipes[version] = pipe
            print(f"[HeartMuLa] Gen Pipeline Ready.")
            
        return self._gen_pipes[version]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            print(f"[HeartMuLa] Loading Transcription Pipeline...")
            from heartlib import HeartTranscriptorPipeline
            
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.float16,
            )
            print("[HeartMuLa] Transcription Pipeline Ready.")
            
        return self._transcribe_pipe

class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                "version": (["3B", "7B"], {"default": "3B"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_audio_length_ms": ("INT", {"default": 240000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, seed, max_audio_length_ms, topk, temperature, cfg_scale, keep_model_loaded):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        try:
            with torch.inference_mode():
                pipe(
                    {"lyrics": lyrics, "tags": tags},
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=out_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    keep_model_loaded=keep_model_loaded
                )
        except Exception as e:
            print(f"[HeartMuLa Error] Generation failed: {e}")
            raise e
        finally:
            if not keep_model_loaded:
                torch.cuda.empty_cache()
                gc.collect()

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
        
        waveform = waveform.to(torch.float32).cpu()
        
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"hm_trans_{uuid.uuid4().hex}.wav")

        torchaudio.save(temp_path, waveform, sr)

        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except:
            temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        try:
            with torch.inference_mode():
                result = pipe(
                    temp_path,
                    temperature=temp_tuple,
                    no_speech_threshold=no_speech_threshold,
                    logprob_threshold=logprob_threshold,
                    task="transcribe",
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            torch.cuda.empty_cache()
            gc.collect()

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