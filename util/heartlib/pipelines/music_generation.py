from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
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
        self._parallel_number = None
        self._muq_dim = None

        if audio_codec is not None:
            self._parallel_number = audio_codec.config.num_quantizers + 1
        elif num_quantizers is not None:
            self._parallel_number = num_quantizers + 1
        else:
            self._parallel_number = 8 + 1

        if model is not None:
            self._muq_dim = model.config.muq_dim

    def load_heartmula(self) -> None:
        if self.model is not None: return
        self.model = HeartMuLa.from_pretrained(self.heartmula_path, dtype=self.dtype, quantization_config=self.bnb_config)
        self.model.to(self.device)
        self.model.eval()
        self._muq_dim = self.model.config.muq_dim

    def unload_heartmula(self) -> None:
        if self.model is None: return
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_heartcodec(self) -> None:
        if self.audio_codec is not None: return
        self.audio_codec = HeartCodec.from_pretrained(self.heartcodec_path, device_map=self.device)
        self._parallel_number = self.audio_codec.config.num_quantizers + 1

    def unload_heartcodec(self) -> None:
        if self.audio_codec is None: return
        del self.audio_codec
        self.audio_codec = None
        gc.collect()
        torch.cuda.empty_cache()

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
        if self._muq_dim is None and self.model is None:
            self.load_heartmula()
        
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp: tags = fp.read()
        tags = tags.lower()
        if not tags.startswith("<tag>"): tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"): tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id: tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id: tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype, device=self.device)
        muq_idx = len(tags_ids)

        lyrics = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp: lyrics = fp.read()
        lyrics_ids = self.text_tokenizer.encode(lyrics.lower()).ids
        if lyrics_ids[0] != self.config.text_bos_id: lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id: lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long, device=self.device)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids, device=self.device)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids, device=self.device)
        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool, device=self.device)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1
        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0: tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long, device=self.device), cfg_scale),
        }

    def _forward(self, model_inputs: Dict[str, Any], max_audio_length_ms: int, temperature: float, topk: int, cfg_scale: float, auto_unload: bool = True):
        self.load_heartmula()
        prompt_tokens, prompt_tokens_mask = model_inputs["tokens"], model_inputs["tokens_mask"]
        continuous_segment, starts, prompt_pos = model_inputs["muq_embed"], model_inputs["muq_idx"], model_inputs["pos"]

        frames = []
        self.model.setup_caches(2 if cfg_scale != 1.0 else 1)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            curr_token = self.model.generate_frame(
                tokens=prompt_tokens, tokens_mask=prompt_tokens_mask, input_pos=prompt_pos,
                temperature=temperature, topk=topk, cfg_scale=cfg_scale,
                continuous_segments=continuous_segment, starts=starts,
            )
        frames.append(curr_token[0:1,])

        max_audio_frames = max_audio_length_ms // 80
        for i in tqdm(range(max_audio_frames)):
            padded_token = (torch.ones((curr_token.shape[0], self._parallel_number), device=curr_token.device, dtype=torch.long) * self.config.empty_id)
            padded_token[:, :-1] = curr_token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(padded_token, device=curr_token.device, dtype=torch.bool)
            padded_token_mask[..., -1] = False

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                curr_token = self.model.generate_frame(
                    tokens=padded_token, tokens_mask=padded_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature, topk=topk, cfg_scale=cfg_scale,
                )
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id): break
            frames.append(curr_token[0:1,])
        
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        
        if auto_unload:
            self.unload_heartmula()
            
        return {"frames": frames.cpu()}

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str, auto_unload: bool = True):
        self.load_heartcodec()
        frames = model_outputs["frames"].to(self.device)
        wav = self.audio_codec.detokenize(frames)
        
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32).cpu()
        
        torchaudio.save(save_path, wav, 48000)
        
        if auto_unload:
            self.unload_heartcodec()

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        auto_unload = kwargs.get("auto_unload", True)
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, cfg_scale=preprocess_kwargs["cfg_scale"])
        # Pass auto_unload to forward and postprocess
        model_outputs = self._forward(model_inputs, auto_unload=auto_unload, **forward_kwargs)
        self.postprocess(model_outputs, save_path=postprocess_kwargs["save_path"], auto_unload=auto_unload)

    def _sanitize_parameters(self, **kwargs):
        return (
            {"cfg_scale": kwargs.get("cfg_scale", 1.5)},
            {"max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000), "temperature": kwargs.get("temperature", 1.0), "topk": kwargs.get("topk", 50), "cfg_scale": kwargs.get("cfg_scale", 1.5)},
            {"save_path": kwargs.get("save_path", "output.mp3")}
        )

    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: torch.device, dtype: torch.dtype, version: str, bnb_config: Optional[BitsAndBytesConfig] = None, lazy_load: bool = True):
        heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
        heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        with open(os.path.join(heartcodec_path, "config.json"), encoding="utf-8") as f:
            num_quantizers = json.load(f).get("num_quantizers", 8)
        tokenizer = Tokenizer.from_file(os.path.join(pretrained_path, "tokenizer.json"))
        gen_config = HeartMuLaGenConfig.from_file(os.path.join(pretrained_path, "gen_config.json"))

        if lazy_load:
            return cls(None, None, None, tokenizer, gen_config, device, dtype, heartmula_path, heartcodec_path, bnb_config, num_quantizers)
        else:
            heartcodec = HeartCodec.from_pretrained(heartcodec_path, device_map=device)
            heartmula = HeartMuLa.from_pretrained(heartmula_path, dtype=dtype, quantization_config=bnb_config)
            return cls(heartmula, heartcodec, None, tokenizer, gen_config, device, dtype, heartmula_path, heartcodec_path, bnb_config, num_quantizers)