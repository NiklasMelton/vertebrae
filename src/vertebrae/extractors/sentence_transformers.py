"""Optional sentence-transformers extractor."""

from typing import Any, Dict, Optional

import numpy as np


class SentenceTransformerExtractor:
    def __init__(
        self,
        name: str,
        model_id: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.modality = "text"
        self.extractor_type = "sentence_transformer"
        self._model = None

    def fit(self, X: Any, y: Any = None) -> "SentenceTransformerExtractor":
        self._load_model()
        return self

    def transform(self, X: Any) -> np.ndarray:
        model = self._load_model()
        texts = list(X)
        return np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerExtractor requires sentence-transformers. "
                    "Install it with: poetry install -E hf"
                ) from exc
            if self.device is None:
                self._model = SentenceTransformer(self.model_id)
            else:
                self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model
