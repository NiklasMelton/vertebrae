"""Embedding compression adapters and utilities."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.config import EmbeddingCompressionConfig
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


@dataclass
class CompressionResult:
    """Compressed embedding matrix plus serializable metadata."""

    embeddings: Any
    metadata: Dict[str, Any]


class EmbeddingCompressor:
    """Protocol-style base for embedding compressors."""

    def fit(self, Z: Any, y: Any = None) -> "EmbeddingCompressor":
        raise NotImplementedError

    def transform(self, Z: Any) -> Any:
        raise NotImplementedError

    def fit_transform(self, Z: Any, y: Any = None) -> Any:
        self.fit(Z, y=y)
        return self.transform(Z)

    def recipe(self) -> Dict[str, Any]:
        raise NotImplementedError


class _IdentityCompressor(EmbeddingCompressor):
    def __init__(self, config: EmbeddingCompressionConfig) -> None:
        self.config = config

    def fit(self, Z: Any, y: Any = None) -> "_IdentityCompressor":
        return self

    def transform(self, Z: Any) -> Any:
        return ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)

    def recipe(self) -> Dict[str, Any]:
        return {
            "method": "none",
            "enabled": bool(self.config.enabled),
        }


class _PrefixTruncateCompressor(EmbeddingCompressor):
    def __init__(self, config: EmbeddingCompressionConfig) -> None:
        self.config = config

    def fit(self, Z: Any, y: Any = None) -> "_PrefixTruncateCompressor":
        return self

    def transform(self, Z: Any) -> Any:
        matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        transformed = matrix[:, : self.config.n_components]
        return _validate_compression_output(
            transformed,
            "prefix_truncate compressed embeddings",
            allow_sparse=True,
            dtype=self.config.dtype,
        )

    def recipe(self) -> Dict[str, Any]:
        return {
            "method": self.config.method,
            "n_components": self.config.n_components,
            "assume_matryoshka": self.config.assume_matryoshka,
            "dtype": self.config.dtype,
        }


class _SklearnEmbeddingCompressor(EmbeddingCompressor):
    def __init__(self, config: EmbeddingCompressionConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None

    def fit(self, Z: Any, y: Any = None) -> "_SklearnEmbeddingCompressor":
        matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        if (
            self.config.method in {"pca", "incremental_pca"}
            and self.config.n_components is not None
            and self.config.n_components > matrix.shape[0]
        ):
            raise ValueError(
                f"Compression method '{self.config.method}' cannot fit n_components="
                f"{self.config.n_components} from only {matrix.shape[0]} fit-side samples; "
                "reduce n_components or provide more fit-side samples."
            )
        if self.config.method in {"pca", "incremental_pca"} and is_sparse_matrix(matrix):
            raise ValueError(
                f"Compression method '{self.config.method}' requires dense embeddings; "
                "use truncated_svd or a random projection for sparse inputs."
            )
        self._model = _build_model(self.config, matrix)
        self._model.fit(matrix)
        return self

    def transform(self, Z: Any) -> np.ndarray:
        if self._model is None:
            raise ValueError("Embedding compressor must be fitted before transform().")
        matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        transformed = self._model.transform(matrix)
        return _validate_compression_output(
            transformed,
            f"{self.config.method} compressed embeddings",
            allow_sparse=False,
            dtype=self.config.dtype,
        )

    def recipe(self) -> Dict[str, Any]:
        return {
            "method": self.config.method,
            "n_components": self.config.n_components,
            "preserve_variance": self.config.preserve_variance,
            "random_state": self.config.random_state,
            "whiten": self.config.whiten,
            "dtype": self.config.dtype,
            "algorithm_kwargs": dict(self.config.algorithm_kwargs),
        }

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ValueError("Embedding compressor has not been fitted.")
        return self._model


class _QuantizeCompressor(EmbeddingCompressor):
    def __init__(self, config: EmbeddingCompressionConfig) -> None:
        self.config = config
        self._calibration: Dict[str, Any] = {}

    def fit(self, Z: Any, y: Any = None) -> "_QuantizeCompressor":
        matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        precision = self.config.precision
        if precision in {"int8", "uint8"} and is_sparse_matrix(matrix):
            raise ValueError(f"Compression precision '{precision}' requires dense embeddings.")
        if precision == "int8":
            dense = _validate_compression_output(
                matrix,
                "quantize calibration embeddings",
                allow_sparse=False,
                dtype=np.float32,
            )
            scale = np.max(np.abs(dense), axis=0)
            scale[scale == 0.0] = 1.0
            self._calibration = {
                "mode": "symmetric_absmax",
                "scale": scale,
                "encoded_dtype": "int8",
                "scoring_dtype": "float32",
            }
        elif precision == "uint8":
            dense = _validate_compression_output(
                matrix,
                "quantize calibration embeddings",
                allow_sparse=False,
                dtype=np.float32,
            )
            min_values = np.min(dense, axis=0)
            max_values = np.max(dense, axis=0)
            ranges = max_values - min_values
            ranges[ranges == 0.0] = 1.0
            self._calibration = {
                "mode": "affine_minmax",
                "min_values": min_values,
                "ranges": ranges,
                "encoded_dtype": "uint8",
                "scoring_dtype": "float32",
            }
        else:
            sparse_input = is_sparse_matrix(matrix)
            self._calibration = {
                "mode": "cast_round_trip" if sparse_input else "cast",
                "encoded_dtype": "float16",
                "scoring_dtype": "float32" if sparse_input else "float16",
            }
        return self

    def transform(self, Z: Any) -> Any:
        matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        precision = self.config.precision
        if precision == "float16":
            if is_sparse_matrix(matrix):
                return _sparse_float16_round_trip(
                    matrix,
                    "quantize compressed embeddings",
                )
            return _validate_compression_output(
                matrix,
                "quantize compressed embeddings",
                allow_sparse=True,
                dtype=np.float16,
            )
        dense = _validate_compression_output(
            matrix,
            "quantize compressed embeddings",
            allow_sparse=False,
            dtype=np.float32,
        )
        if precision == "int8":
            scale = self._calibration["scale"]
            encoded = np.clip(np.rint((dense / scale) * 127.0), -127, 127).astype(np.int8)
            transformed = (encoded.astype(np.float32) * scale) / 127.0
        else:
            min_values = self._calibration["min_values"]
            ranges = self._calibration["ranges"]
            encoded = np.clip(
                np.rint(((dense - min_values) / ranges) * 255.0),
                0,
                255,
            ).astype(np.uint8)
            transformed = (encoded.astype(np.float32) * ranges / 255.0) + min_values
        return _validate_compression_output(
            transformed,
            "quantize compressed embeddings",
            allow_sparse=False,
        )

    def recipe(self) -> Dict[str, Any]:
        return {
            "method": self.config.method,
            "precision": self.config.precision,
            "dtype": self.config.dtype,
        }

    @property
    def calibration(self) -> Dict[str, Any]:
        return self._calibration


def _validate_compression_output(
    value: Any,
    name: str,
    *,
    allow_sparse: bool,
    dtype: Any = None,
) -> Any:
    """Validate a compression-stage matrix after dtype or precision conversion."""

    if dtype is not None:
        resolved_dtype = np.dtype(dtype)
        if is_sparse_matrix(value) and resolved_dtype == np.dtype(np.float16):
            raise ValueError(
                f"{name} cannot use dtype='float16' while preserving sparse storage "
                "because scipy.sparse does not portably support float16. Use "
                "dtype='float32' or method='quantize' with precision='float16'."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(resolved_dtype, copy=False)
    matrix = ensure_numeric_matrix(value, name, allow_sparse=allow_sparse)
    if not np.issubdtype(matrix.dtype, np.floating):
        if np.issubdtype(matrix.dtype, np.integer):
            matrix = matrix.astype(float, copy=False)
        else:
            raise ValueError(f"{name} must use a real floating-point dtype.")
    return matrix


def _sparse_float16_round_trip(value: Any, name: str) -> Any:
    """Apply float16 rounding to sparse data while retaining a supported dtype."""

    matrix = ensure_numeric_matrix(value, name, allow_sparse=True)
    transformed = matrix.astype(np.float32, copy=True).tocsr(copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed.data = transformed.data.astype(np.float16).astype(np.float32)
    return _validate_compression_output(transformed, name, allow_sparse=True)


def compress_embeddings(
    Z: Any,
    config: Optional[EmbeddingCompressionConfig] = None,
    y: Any = None,
) -> CompressionResult:
    """Compress embeddings according to a compression config."""

    compression_config = config or EmbeddingCompressionConfig()
    matrix = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
    original_dim = int(matrix.shape[1])
    original_sparse = is_sparse_matrix(matrix)
    if not compression_config.enabled or compression_config.method == "none":
        metadata = {
            "enabled": False,
            "method": "none",
            "applied": False,
            "original_dim": original_dim,
            "compressed_dim": original_dim,
            "input_sparse": original_sparse,
            "output_sparse": original_sparse,
            "dtype": str(getattr(matrix, "dtype", "unknown")),
            "warnings": [],
            "recipe": {"method": "none", "enabled": False},
        }
        return CompressionResult(embeddings=matrix, metadata=metadata)

    compressor = create_embedding_compressor(compression_config)
    warnings = []
    target_dim = compression_config.n_components
    if target_dim is not None and target_dim >= original_dim:
        warnings.append(
            "Requested compression dimension is greater than or equal to the original "
            "embedding dimension; skipping compression."
        )
        metadata = {
            "enabled": True,
            "method": compression_config.method,
            "applied": False,
            "original_dim": original_dim,
            "compressed_dim": original_dim,
            "input_sparse": original_sparse,
            "output_sparse": original_sparse,
            "dtype": str(getattr(matrix, "dtype", "unknown")),
            "warnings": warnings,
            "recipe": compressor.recipe(),
        }
        return CompressionResult(embeddings=matrix, metadata=metadata)

    transformed = compressor.fit_transform(matrix, y=y)
    metadata = _compression_metadata(
        compressor=compressor,
        original=matrix,
        transformed=transformed,
        warnings=warnings,
    )
    return CompressionResult(embeddings=transformed, metadata=metadata)


def create_embedding_compressor(
    config: Optional[EmbeddingCompressionConfig] = None,
) -> EmbeddingCompressor:
    compression_config = config or EmbeddingCompressionConfig()
    if not compression_config.enabled or compression_config.method == "none":
        return _IdentityCompressor(compression_config)
    if compression_config.method == "prefix_truncate":
        return _PrefixTruncateCompressor(compression_config)
    if compression_config.method == "quantize":
        return _QuantizeCompressor(compression_config)
    return _SklearnEmbeddingCompressor(compression_config)


def compression_recipe_hash(config: EmbeddingCompressionConfig) -> str:
    return hash_json_exact({"identity_schema": 2, "compression_config": asdict(config)})


def compress_embedding_artifact_key(
    embedding_key: str,
    config: EmbeddingCompressionConfig,
) -> str:
    return f"{embedding_key}/compressions/{compression_recipe_hash(config)}"


def _compression_metadata(
    compressor: EmbeddingCompressor,
    original: Any,
    transformed: Any,
    warnings: list[str],
) -> Dict[str, Any]:
    output_sparse = is_sparse_matrix(transformed)
    original_bytes = _matrix_nbytes(original)
    transformed_bytes = _matrix_nbytes(transformed)
    metadata_warnings = list(warnings)
    metadata = {
        "enabled": True,
        "method": compressor.recipe().get("method"),
        "applied": True,
        "original_dim": int(original.shape[1]),
        "compressed_dim": int(transformed.shape[1]),
        "input_sparse": is_sparse_matrix(original),
        "output_sparse": output_sparse,
        "dtype": str(getattr(transformed, "dtype", "unknown")),
        "resident_bytes": transformed_bytes,
        "compression_ratio": (
            float(original_bytes) / float(transformed_bytes)
            if transformed_bytes not in {0, None}
            else None
        ),
        "warnings": metadata_warnings,
        "recipe": compressor.recipe(),
    }
    if isinstance(compressor, _PrefixTruncateCompressor):
        metadata["assume_matryoshka"] = compressor.config.assume_matryoshka
        if not compressor.config.assume_matryoshka:
            metadata_warnings.append(
                "Prefix truncation was applied without assume_matryoshka=True; "
                "interpret this as a generic dimension-prefix diagnostic."
            )
    if isinstance(compressor, _QuantizeCompressor):
        calibration = dict(compressor.calibration)
        encoded_bytes = _estimated_quantized_bytes(
            original,
            compressor.config.precision,
        )
        metadata["precision"] = compressor.config.precision
        metadata["quantization_mode"] = calibration.get("mode")
        metadata["encoded_dtype"] = calibration.get("encoded_dtype")
        metadata["scoring_dtype"] = calibration.get("scoring_dtype")
        metadata["calibration"] = make_json_safe(calibration)
        metadata["estimated_encoded_bytes"] = encoded_bytes
        if encoded_bytes:
            metadata["compression_ratio"] = float(original_bytes) / float(encoded_bytes)
    sklearn_model = getattr(compressor, "model", None)
    if sklearn_model is not None:
        if hasattr(sklearn_model, "explained_variance_ratio_"):
            values = sklearn_model.explained_variance_ratio_
            metadata["explained_variance_ratio"] = make_json_safe(np.asarray(values))
            metadata["explained_variance_total"] = float(np.sum(values))
        if hasattr(sklearn_model, "n_components_"):
            metadata["resolved_n_components"] = int(sklearn_model.n_components_)
    return metadata


def _build_model(config: EmbeddingCompressionConfig, matrix: Any) -> Any:
    method = config.method
    kwargs = dict(config.algorithm_kwargs)
    if method == "pca":
        from sklearn.decomposition import PCA

        n_components: Any = config.preserve_variance
        if config.n_components is not None:
            n_components = config.n_components
        return PCA(
            n_components=n_components,
            whiten=config.whiten,
            random_state=config.random_state,
            **kwargs,
        )
    if method == "incremental_pca":
        from sklearn.decomposition import IncrementalPCA

        return IncrementalPCA(
            n_components=config.n_components,
            whiten=config.whiten,
            **kwargs,
        )
    if method == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD(
            n_components=config.n_components,
            random_state=config.random_state,
            **kwargs,
        )
    if method == "gaussian_random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        return GaussianRandomProjection(
            n_components=config.n_components,
            random_state=config.random_state,
            **kwargs,
        )
    if method == "sparse_random_projection":
        from sklearn.random_projection import SparseRandomProjection

        return SparseRandomProjection(
            n_components=config.n_components,
            random_state=config.random_state,
            **kwargs,
        )
    raise ValueError(f"Unsupported compression method: {method!r}.")


def _matrix_nbytes(value: Any) -> int:
    if is_sparse_matrix(value):
        return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    return int(np.asarray(value).nbytes)


def _estimated_quantized_bytes(value: Any, precision: Optional[str]) -> Optional[int]:
    if precision is None:
        return None
    if is_sparse_matrix(value):
        sparse = value
        if precision == "float16":
            return int(
                sparse.data.astype(np.float16).nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
            )
        return None
    dense = np.asarray(value)
    rows, cols = dense.shape
    if precision == "float16":
        return int(rows * cols * np.dtype(np.float16).itemsize)
    if precision == "int8":
        return int(rows * cols * np.dtype(np.int8).itemsize)
    if precision == "uint8":
        return int(rows * cols * np.dtype(np.uint8).itemsize)
    return None
