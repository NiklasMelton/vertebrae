"""Artifact store factory helpers."""

from typing import Any
from urllib.parse import urlparse

from vertebrae.cache.artifact_store import ArtifactStore, ArtifactStoreConfig
from vertebrae.cache.local_store import LocalArtifactStore


def create_artifact_store(uri: str, **options: Any) -> ArtifactStore:
    """Create an artifact store from a local path or URI.

    Args:
        uri: Local cache path or provider URI such as `s3://...` or `gs://...`.
        **options: Provider-specific construction options.

    Returns:
        Configured artifact store.

    Raises:
        ValueError: If the URI scheme is unsupported.
    """

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else uri
        return LocalArtifactStore(path)
    if parsed.scheme == "s3":
        from vertebrae.cache.s3_store import S3ArtifactStore

        return S3ArtifactStore.from_uri(uri, **options)
    if parsed.scheme == "gs":
        from vertebrae.cache.gcs_store import GCSArtifactStore

        return GCSArtifactStore.from_uri(uri, **options)
    raise ValueError(f"Unsupported artifact store URI: {uri}.")


def create_artifact_store_from_config(config: ArtifactStoreConfig) -> ArtifactStore:
    """Create an artifact store from a serialized store config."""

    return create_artifact_store(config.uri, **config.options)
