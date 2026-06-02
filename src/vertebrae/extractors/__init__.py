"""Feature extractor implementations."""

from vertebrae.extractors.callable import CallableExtractor
from vertebrae.extractors.huggingface_text import HFTextExtractor
from vertebrae.extractors.precomputed import PrecomputedExtractor
from vertebrae.extractors.sentence_transformers import SentenceTransformerExtractor
from vertebrae.extractors.sklearn import SklearnExtractor

__all__ = [
    "CallableExtractor",
    "HFTextExtractor",
    "PrecomputedExtractor",
    "SentenceTransformerExtractor",
    "SklearnExtractor",
]
