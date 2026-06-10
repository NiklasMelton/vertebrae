"""Feature extractor implementations."""

from vertebrae.extractors.callable import CallableExtractor
from vertebrae.extractors.huggingface_audio import HFAudioExtractor
from vertebrae.extractors.huggingface_text import HFTextExtractor
from vertebrae.extractors.huggingface_time_series import HFTimeSeriesExtractor
from vertebrae.extractors.huggingface_vision import HFVisionExtractor
from vertebrae.extractors.keras import KerasExtractor
from vertebrae.extractors.multi_output import MultiOutputExtractor
from vertebrae.extractors.onnx import ONNXExtractor
from vertebrae.extractors.precomputed import PrecomputedExtractor
from vertebrae.extractors.sentence_transformers import SentenceTransformerExtractor
from vertebrae.extractors.sklearn import SklearnExtractor
from vertebrae.extractors.torch import TorchExtractor

__all__ = [
    "CallableExtractor",
    "HFAudioExtractor",
    "KerasExtractor",
    "HFTextExtractor",
    "HFTimeSeriesExtractor",
    "HFVisionExtractor",
    "MultiOutputExtractor",
    "ONNXExtractor",
    "PrecomputedExtractor",
    "SentenceTransformerExtractor",
    "SklearnExtractor",
    "TorchExtractor",
]
