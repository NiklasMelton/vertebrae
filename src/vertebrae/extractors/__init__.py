"""Feature extractor implementations."""

from vertebrae.extractors.callable import CallableExtractor
from vertebrae.extractors.huggingface_text import HFTextExtractor
from vertebrae.extractors.huggingface_vision import HFVisionExtractor
from vertebrae.extractors.keras import KerasExtractor
from vertebrae.extractors.onnx import ONNXExtractor
from vertebrae.extractors.precomputed import PrecomputedExtractor
from vertebrae.extractors.sentence_transformers import SentenceTransformerExtractor
from vertebrae.extractors.sklearn import SklearnExtractor
from vertebrae.extractors.torch import TorchExtractor

__all__ = [
    "CallableExtractor",
    "KerasExtractor",
    "HFTextExtractor",
    "HFVisionExtractor",
    "ONNXExtractor",
    "PrecomputedExtractor",
    "SentenceTransformerExtractor",
    "SklearnExtractor",
    "TorchExtractor",
]
