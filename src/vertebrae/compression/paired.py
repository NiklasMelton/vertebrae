"""Shared fitted compression for paired embedding endpoints."""

from typing import Any, Dict, Tuple

from vertebrae.compression.base import (
    _compression_metadata,
    compress_embeddings,
    create_embedding_compressor,
)
from vertebrae.config import EmbeddingCompressionConfig
from vertebrae.utils.validation import ensure_numeric_matrix


def compress_embedding_pair(
    samples: Any,
    prompts: Any,
    config: EmbeddingCompressionConfig,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fit label-free compression on samples and apply it to paired prompts."""

    sample_matrix = ensure_numeric_matrix(samples, "sample embeddings", allow_sparse=True)
    prompt_matrix = ensure_numeric_matrix(prompts, "prompt embeddings", allow_sparse=True)
    if sample_matrix.shape[1] != prompt_matrix.shape[1]:
        raise ValueError("Paired embeddings must have the same feature dimension.")
    if not isinstance(config, EmbeddingCompressionConfig):
        raise TypeError("config must be an EmbeddingCompressionConfig.")

    if not config.enabled or config.method == "none":
        result = compress_embeddings(sample_matrix, config=config)
        return result.embeddings, prompt_matrix, dict(result.metadata)

    if config.n_components is not None and config.n_components >= sample_matrix.shape[1]:
        result = compress_embeddings(sample_matrix, config=config)
        return result.embeddings, prompt_matrix, dict(result.metadata)

    compressor = create_embedding_compressor(config)
    compressed_samples = compressor.fit_transform(sample_matrix)
    compressed_prompts = compressor.transform(prompt_matrix)
    metadata = _compression_metadata(compressor, sample_matrix, compressed_samples, warnings=[])
    return compressed_samples, compressed_prompts, metadata
