from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
import comfy.utils
import torchaudio
import soundfile as sf
import json
import gc
from transformers import BitsAndBytesConfig

@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)

class HeartMuLaGenPipeline:
    def __init__(
        self,
        model: Optional[HeartMuLa],
        audio_codec: Optional[HeartCodec],
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
        heartmula_path: Optional[str] = None,
        heartcodec_path: Optional[str] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        num_quantizers: Optional[int] = None,
    ):
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.heartmula_path = heartmula_path
        self.heartcodec_path = heartcodec_path
        self.bnb_config = bnb_config
        self._parallel_number = num_quantizers + 1 if num_quantizers else 9
        self._muq_dim = model.config.muq_dim if model else None

    def load_heartmula(self):
        if self.model is None:
            self.model = HeartMuLa.from_pretrained(
                self.heartmula_path, 
                torch_dtype=self.dtype, 
                quantization_config=self.bnb_config
            )
        if str(next(self.model.parameters()).device) != str(self.device):
            self.model.to(self.device)
        self.model.eval()
        self._muq_dim = self.model.config.muq_dim

    def load_heartcodec(self):
        if self.audio_codec is None:
            self.audio_codec = HeartCodec.from_pretrained(self.heartcodec_path)
        if str(next(self.audio_codec.parameters()).device) != str(self.device):
            self.audio_codec.to(self.device)
        self.audio_codec.eval()

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
        self.load_heartmula()
        tags = inputs["tags"].lower()
        if not tags.startswith("<tag>"): tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"): tags = f"{tags}</tag>"
        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id: tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id: tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype, device=self.device)
        muq_idx = len(tags_ids)

        lyrics = inputs["lyrics"].lower()
        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id: lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id: lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long, device=self.device)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids, device=self.device)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids, device=self.device)
        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool, device=self.device)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1
        def _cfg_cat(t):
            t = t.unsqueeze(0)
            return torch.cat([t, t], dim=0) if cfg_scale != 1.0 else t

        return {
            "tokens": _cfg_cat(tokens),
            "tokens_mask": _cfg_cat(tokens_mask),
            "muq_embed": _cfg_cat(muq_embed),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long, device=self.device)),
        }

    def _forward(self, model_inputs: Dict[str, Any], max_audio_length_ms: int, temperature: float, topk: int, cfg_scale: float):
        self.load_heartmula()
        self.model.setup_caches(2 if cfg_scale != 1.0 else 1)
        
        frames = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            curr_token = self.model.generate_frame(
                tokens=model_inputs["tokens"], tokens_mask=model_inputs["tokens_mask"], 
                input_pos=model_inputs["pos"], temperature=temperature, topk=topk, 
                cfg_scale=cfg_scale, continuous_segments=model_inputs["muq_embed"], starts=model_inputs["muq_idx"],
            )
        frames.append(curr_token[0:1,])

        max_frames = max_audio_length_ms // 80
        pbar = comfy.utils.ProgressBar(max_frames)
        for i in range(max_frames):
            padded_token = (torch.ones((curr_token.shape[0], self._parallel_number), device=self.device, dtype=torch.long) * self.config.empty_id)
            padded_token[:, :-1] = curr_token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(padded_token, dtype=torch.bool); padded_token_mask[..., -1] = False

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                curr_token = self.model.generate_frame(
                    tokens=padded_token, tokens_mask=padded_token_mask,
                    input_pos=model_inputs["pos"][..., -1:] + i + 1,
                    temperature=temperature, topk=topk, cfg_scale=cfg_scale,
                )
            pbar.update(1)
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id): break
            frames.append(curr_token[0:1,])
        
        return torch.stack(frames).permute(1, 2, 0).squeeze(0).cpu()

    def postprocess(self, frames: torch.Tensor, save_path: str, keep_model_loaded: bool, offload_mode: str = "auto"):
        if offload_mode == "aggressive":
            if self.model is not None:
                del self.model
                self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
        else:
            if self.model is not None:
                self.model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        try:
            self.load_heartcodec()
            with torch.inference_mode():
                wav = self.audio_codec.detokenize(frames.to(self.device))
                wav = wav.detach().cpu().float()
            
            try:
                torchaudio.save(save_path, wav, 48000)
            except Exception:
                wav_np = wav.numpy()
                if wav_np.ndim == 2:
                    wav_np = wav_np.T
                sf.write(save_path, wav_np, 48000)
        finally:
            if hasattr(self, 'audio_codec'):
                del self.audio_codec
                self.audio_codec = None

            torch.cuda.empty_cache()
            gc.collect()

            if keep_model_loaded and offload_mode != "aggressive":
                if self.model is not None:
                    self.model.to(self.device)
            else:
                if self.model is not None:
                    del self.model
                    self.model = None
            torch.cuda.empty_cache()

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        keep_model_loaded = kwargs.get("keep_model_loaded", True)
        offload_mode = kwargs.get("offload_mode", "auto")
        model_inputs = self.preprocess(inputs, cfg_scale=kwargs.get("cfg_scale", 1.5))
        frames = self._forward(model_inputs,
                               max_audio_length_ms=kwargs.get("max_audio_length_ms", 120000),
                               temperature=kwargs.get("temperature", 1.0),
                               topk=kwargs.get("topk", 50),
                               cfg_scale=kwargs.get("cfg_scale", 1.5))
        self.postprocess(frames, kwargs.get("save_path", "out.wav"), keep_model_loaded, offload_mode)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: torch.device, torch_dtype: torch.dtype, version: str, bnb_config=None, lazy_load=True):
        heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
        heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        tokenizer = Tokenizer.from_file(os.path.join(pretrained_path, "tokenizer.json"))
        gen_config = HeartMuLaGenConfig.from_file(os.path.join(pretrained_path, "gen_config.json"))
        return cls(None, None, None, tokenizer, gen_config, device, torch_dtype, heartmula_path, heartcodec_path, bnb_config)