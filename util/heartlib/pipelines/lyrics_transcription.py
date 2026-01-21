from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.processing_whisper import WhisperProcessor
import torch
import os


class HeartTranscriptorPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_path: str, device: torch.device, dtype: torch.dtype
    ):
        if os.path.exists(
            hearttranscriptor_path := os.path.join(
                pretrained_path, "HeartTranscriptor-oss"
            )
        ):
            model = WhisperForConditionalGeneration.from_pretrained(
                hearttranscriptor_path, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            processor = WhisperProcessor.from_pretrained(hearttranscriptor_path)
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartTranscriptor at {hearttranscriptor_path} but not found. Please check your folder {pretrained_path}."
            )

        return cls(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            dtype=dtype,
            chunk_length_s=30,
            batch_size=16,
        )
