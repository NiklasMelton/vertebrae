"""Feature extractor implementations."""

from vertebrae.extractors.callable import CallableExtractor
from vertebrae.extractors.graph import GraphModelExtractor
from vertebrae.extractors.hosted import HostedEmbeddingExtractor
from vertebrae.extractors.huggingface_audio import HFAudioExtractor
from vertebrae.extractors.huggingface_multimodal import HFMultimodalExtractor
from vertebrae.extractors.huggingface_text import HFTextExtractor
from vertebrae.extractors.huggingface_time_series import HFTimeSeriesExtractor
from vertebrae.extractors.huggingface_video import HFVideoExtractor
from vertebrae.extractors.huggingface_vision import HFVisionExtractor
from vertebrae.extractors.jax_flax import JAXFlaxExtractor
from vertebrae.extractors.keras import KerasExtractor
from vertebrae.extractors.multi_output import MultiOutputExtractor
from vertebrae.extractors.onnx import ONNXExtractor
from vertebrae.extractors.openclip import OpenCLIPExtractor
from vertebrae.extractors.precomputed import PrecomputedExtractor
from vertebrae.extractors.retrieval import CallableRetrievalExtractor
from vertebrae.extractors.sentence_transformers import SentenceTransformerExtractor
from vertebrae.extractors.siglip import SigLIPExtractor
from vertebrae.extractors.sklearn import SklearnExtractor
from vertebrae.extractors.spatial import (
    CallableSpatialExtractor,
    PrecomputedSpatialExtractor,
    SpatialEmbeddingOutput,
    SpatialLayout,
    SpatialOutputSpec,
)
from vertebrae.extractors.structured import (
    CallableStructuredExtractor,
    PrecomputedStructuredExtractor,
    StructuredEmbeddingOutput,
    StructuredOutputSpec,
)
from vertebrae.extractors.tensorflow_hub import TFHubExtractor
from vertebrae.extractors.timm import TimmVisionExtractor
from vertebrae.extractors.torch import TorchExtractor
from vertebrae.extractors.torchvision import TorchvisionVisionExtractor
from vertebrae.extractors.tree_leaf import TreeLeafEmbeddingExtractor

__all__ = [
    "CallableExtractor",
    "GraphModelExtractor",
    "HFAudioExtractor",
    "HFMultimodalExtractor",
    "HostedEmbeddingExtractor",
    "KerasExtractor",
    "HFTextExtractor",
    "HFTimeSeriesExtractor",
    "HFVideoExtractor",
    "HFVisionExtractor",
    "JAXFlaxExtractor",
    "MultiOutputExtractor",
    "ONNXExtractor",
    "OpenCLIPExtractor",
    "PrecomputedExtractor",
    "SentenceTransformerExtractor",
    "SigLIPExtractor",
    "SklearnExtractor",
    "TFHubExtractor",
    "TimmVisionExtractor",
    "TorchExtractor",
    "TorchvisionVisionExtractor",
    "TreeLeafEmbeddingExtractor",
    "CallableSpatialExtractor",
    "CallableStructuredExtractor",
    "PrecomputedSpatialExtractor",
    "PrecomputedStructuredExtractor",
    "SpatialEmbeddingOutput",
    "SpatialLayout",
    "SpatialOutputSpec",
    "StructuredEmbeddingOutput",
    "StructuredOutputSpec",
    "CallableRetrievalExtractor",
]
