"""Dataset abstractions."""

from vertebrae.datasets.base import (
    BenchmarkDataset,
    EmbeddingUnitDataset,
    TargetView,
    UnitAnnotation,
)
from vertebrae.datasets.retrieval import RetrievalDataset
from vertebrae.datasets.segmentation import SegmentationAnnotation, SegmentationDataset
from vertebrae.datasets.zero_shot import ZeroShotClassSpec, ZeroShotDataset

__all__ = [
    "BenchmarkDataset",
    "EmbeddingUnitDataset",
    "TargetView",
    "UnitAnnotation",
    "SegmentationAnnotation",
    "SegmentationDataset",
    "RetrievalDataset",
    "ZeroShotClassSpec",
    "ZeroShotDataset",
]
