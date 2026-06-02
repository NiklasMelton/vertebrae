"""Optional sentence-transformers extractor."""

from typing import Any, Dict, Optional

import numpy as np

from vertebrae.utils.validation import ensure_dense_numeric_2d


class SentenceTransformerExtractor:
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
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.modality = "text"
        self.extractor_type = "frozen_pretrained"
        self._model: Any = None

    def fit(self, X: Any, y: Any = None) -> "SentenceTransformerExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
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
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
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
        }

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
