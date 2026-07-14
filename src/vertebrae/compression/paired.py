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
    fit_embeddings: Any,
    paired_embeddings: Any,
    config: EmbeddingCompressionConfig,
    *,
    fit_name: str = "fit embeddings",
    paired_name: str = "paired embeddings",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fit compression on one endpoint and apply it to a paired endpoint."""

    if not isinstance(config, EmbeddingCompressionConfig):
        raise TypeError("config must be an EmbeddingCompressionConfig.")
    fit_matrix = ensure_numeric_matrix(fit_embeddings, fit_name, allow_sparse=True)
    paired_matrix = ensure_numeric_matrix(paired_embeddings, paired_name, allow_sparse=True)
    if fit_matrix.shape[1] != paired_matrix.shape[1]:
        raise ValueError("Paired embeddings must have the same feature dimension.")

    if not config.enabled or config.method == "none":
        result = compress_embeddings(fit_matrix, config=config)
        return result.embeddings, paired_matrix, dict(result.metadata)

    if config.n_components is not None and config.n_components >= fit_matrix.shape[1]:
        result = compress_embeddings(fit_matrix, config=config)
        return result.embeddings, paired_matrix, dict(result.metadata)

    compressor = create_embedding_compressor(config)
    compressed_fit = compressor.fit_transform(fit_matrix)
    compressed_paired = compressor.transform(paired_matrix)
    metadata = _compression_metadata(compressor, fit_matrix, compressed_fit, warnings=[])
    return compressed_fit, compressed_paired, metadata
