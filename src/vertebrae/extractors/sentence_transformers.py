"""Optional sentence-transformers extractor."""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from vertebrae.utils.validation import ensure_dense_numeric_2d


class SentenceTransformerExtractor:
    """Sentence-transformers embedding extractor.

    Args:
        name: User-facing extractor name.
        model_id: Sentence-transformers model identifier or local path.
        batch_size: Batch size passed to `model.encode`.
        normalize_embeddings: Whether sentence-transformers should normalize outputs.
        device: Optional device string.
        show_progress_bar: Whether to show sentence-transformers progress output.
        model_kwargs: Extra keyword arguments for `SentenceTransformer`.
        encode_kwargs: Extra keyword arguments for `model.encode`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        show_progress_bar: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_paths: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.modality = "text"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._model: Any = None

    def fit(self, X: Any, y: Any = None) -> "SentenceTransformerExtractor":
        """No-op fit for frozen sentence-transformers models.

        Args:
            X: Input text samples.
            y: Optional labels.

        Returns:
            This extractor.
        """

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode text inputs into dense embeddings.

        Args:
            X: Sequence of strings.

        Returns:
            Dense numeric embedding matrix.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If inputs are not strings or output is invalid.
        """

        model = self._load_model()
        texts = _validate_text_sequence(X, "SentenceTransformerExtractor")
        output = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
            **self.encode_kwargs,
        )
        return ensure_dense_numeric_2d(
            np.asarray(output, dtype=np.float32),
            f"SentenceTransformerExtractor '{self.name}' output",
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode text inputs into dense embeddings.

        Args:
            X: Sequence of strings.
            y: Optional labels.

        Returns:
            Dense numeric embedding matrix.
        """

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable sentence-transformers recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "show_progress_bar": self.show_progress_bar,
            "model_kwargs": self.model_kwargs,
            "encode_kwargs": self.encode_kwargs,
            "streaming_safe": self.streaming_safe,
        }

    def get_resource_profile_adapter(self) -> Any:
        from vertebrae.profiling import TorchResourceProfileAdapter

        def resolve_device(torch: Any) -> Any:
            if self.device is not None:
                return self.device
            if self._model is not None and getattr(self._model, "device", None) is not None:
                return self._model.device
            return "cuda" if torch.cuda.is_available() else "cpu"

        return TorchResourceProfileAdapter(
            self,
            self.checkpoint_paths,
            model_getter=lambda: self._model,
            device_resolver=resolve_device,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerExtractor requires optional Hugging Face "
                    "dependencies. Install with the documented Hugging Face extra or "
                    "Poetry group."
                ) from exc
            if self.device is None:
                self._model = SentenceTransformer(self.model_id, **self.model_kwargs)
            else:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    **self.model_kwargs,
                )
        return self._model


def _validate_text_sequence(value: Any, owner: str) -> list[str]:
    if isinstance(value, str):
        raise ValueError(f"{owner} expects a sequence of strings, not a single string.")
    try:
        texts = list(value)
    except TypeError as exc:
        raise ValueError(f"{owner} expects a sequence of strings.") from exc
    if not all(isinstance(text, str) for text in texts):
        raise ValueError(f"{owner} expects every input item to be a string.")
    return texts
