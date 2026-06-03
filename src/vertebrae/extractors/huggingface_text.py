"""Optional Hugging Face text embedding extractor."""

from typing import Any, Dict, List, Optional

import numpy as np


class HFTextExtractor:
    """Hugging Face text backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        pooling: Pooling mode: `"mean"`, `"cls"`, or `"last_token"`.
        batch_size: Number of texts encoded per batch.
        max_length: Tokenizer truncation length.
        device: Optional device string.
        revision: Optional model revision.
        trust_remote_code: Whether to allow remote model code.
        tokenizer_kwargs: Extra keyword arguments for `AutoTokenizer`.
        model_kwargs: Extra keyword arguments for `AutoModel`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        pooling: str = "mean",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pooling not in {"mean", "cls", "last_token"}:
            raise ValueError("pooling must be one of: mean, cls, last_token.")
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.modality = "text"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def fit(self, X: Any, y: Any = None) -> "HFTextExtractor":
        """No-op fit for frozen Hugging Face text models.

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
            Dense float32 embedding matrix.

        Raises:
            ImportError: If optional Hugging Face dependencies are missing.
            ValueError: If inputs are invalid.
        """

        tokenizer, model, torch = self._load_model()
        texts = _validate_text_sequence(X, "HFTextExtractor")
        outputs: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    **self.tokenizer_kwargs,
                )
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"], torch)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(outputs).astype(np.float32, copy=False) if outputs else np.empty((0, 0))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode text inputs into dense embeddings.

        Args:
            X: Sequence of strings.
            y: Optional labels.

        Returns:
            Dense float32 embedding matrix.
        """

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face text recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "pooling": self.pooling,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "model_kwargs": self.model_kwargs,
            "streaming_safe": self.streaming_safe,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "HFTextExtractor requires optional Hugging Face dependencies. "
                    "Install with the documented Hugging Face extra or Poetry group."
                ) from exc
            common_kwargs = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            common_kwargs = {
                key: value for key, value in common_kwargs.items() if value is not None
            }
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                **common_kwargs,
                **self.tokenizer_kwargs,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **common_kwargs,
                **self.model_kwargs,
            )
            self._torch = torch
            assert self._model is not None
            self._model.to(self._device(torch))
        return self._tokenizer, self._model, self._torch

    def _device(self, torch: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _pool(self, hidden: Any, mask: Any, torch: Any) -> Any:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        if self.pooling == "last_token":
            lengths = mask.sum(dim=1) - 1
            batch_ids = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_ids, lengths, :]
        expanded_mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked_hidden = hidden * expanded_mask
        lengths = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return masked_hidden.sum(dim=1) / lengths


def _validate_text_sequence(value: Any, owner: str) -> List[str]:
    if isinstance(value, str):
        raise ValueError(f"{owner} expects a sequence of strings, not a single string.")
    try:
        texts = list(value)
    except TypeError as exc:
        raise ValueError(f"{owner} expects a sequence of strings.") from exc
    if not all(isinstance(text, str) for text in texts):
        raise ValueError(f"{owner} expects every input item to be a string.")
    return texts
