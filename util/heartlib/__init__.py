from .pipelines.music_generation import HeartMuLaGenPipeline
from .pipelines.lyrics_transcription import HeartTranscriptorPipeline
from .pipelines.optimized_music_generation import (
    OptimizedHeartMuLaGenPipeline,
    OptimizationConfig,
    SectionStyle,
    generate_music_optimized,
)

__all__ = [
    "HeartMuLaGenPipeline",
    "HeartTranscriptorPipeline",
    # Optimized pipeline (based on paper arXiv:2601.10547)
    "OptimizedHeartMuLaGenPipeline",
    "OptimizationConfig",
    "SectionStyle",
    "generate_music_optimized",
]