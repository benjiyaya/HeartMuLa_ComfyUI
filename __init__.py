import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import folder_paths
import gc

# ----------------------------
# 显存环境优化 (全局设置)
# ----------------------------
# 开启可扩展段内存分配，有效减少长音频生成时的显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# Global Model Manager (Singleton)
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
        if version not in self._gen_pipes:
            print(f"[HeartMuLa] Loading Generation Pipeline (Version: {version}) on {self._device}...")
            from heartlib import HeartMuLaGenPipeline
            
            # 采用 BF16 加载，与原始优化脚本保持一致
            pipe = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.bfloat16,
                version=version,
            )
            # 确保子组件也处于 BF16
            pipe.model.to(dtype=torch.bfloat16)
            pipe.audio_codec.to(dtype=torch.bfloat16)
            
            self._gen_pipes[version] = pipe
            
            # 加载后立即清理一次碎片
            torch.cuda.empty_cache()
            gc.collect()
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
            torch.cuda.empty_cache()
            gc.collect()
            print("[HeartMuLa] Transcription Pipeline Ready.")
            
        return self._transcribe_pipe

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
                "max_audio_length_ms": ("INT", {"default": 240000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, max_audio_length_ms, topk, temperature, cfg_scale):
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        # --- 执行显存优化后的生成流程 ---
        try:
            # 使用更高效的 torch.inference_mode 替代 torch.no_grad
            # 使用 torch.autocast 确保在 BF16 下自动混合精度运行
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pipe(
                    {"lyrics": lyrics, "tags": tags},
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=out_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                )
        except Exception as e:
            print(f"[HeartMuLa Error] Generation failed: {e}")
            raise e
        finally:
            # 关键：生成结束后立即释放显存缓存，防止显存阶梯式上涨
            torch.cuda.empty_cache()
            gc.collect()

        # --- 后处理逻辑 ---
        waveform, sample_rate = torchaudio.load(out_path)
        
        # 统一输出格式为 ComfyUI 标准的 [Batch, Channels, Samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        
        if waveform.dtype != torch.float32:
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, audio_input, temperature_tuple, no_speech_threshold, logprob_threshold):
        # 解析输入音频
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
            if isinstance(waveform, np.ndarray):
                 waveform = torch.from_numpy(waveform)

        # 转换为单声道处理以减少转录负担
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        if waveform.shape[0] > 1: # 如果是多声道，取均值转单声道
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
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
            # 转录同样开启推理模式和自动精度
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
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