"""Optional Hugging Face text embedding extractor."""

from typing import Any, Dict, List, Optional

import numpy as np


class HFTextExtractor:
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
        self.modality = "text"
        self.extractor_type = "huggingface_text"
        self._tokenizer = None
        self._model = None
        self._torch = None

    def fit(self, X: Any, y: Any = None) -> "HFTextExtractor":
        self._load_model()
        return self

    def transform(self, X: Any) -> np.ndarray:
        tokenizer, model, torch = self._load_model()
        texts = list(X)
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
                )
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"], torch)
                outputs.append(pooled.detach().cpu().numpy())
        return np.vstack(outputs) if outputs else np.empty((0, 0))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "model_id": self.model_id,
            "pooling": self.pooling,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "HFTextExtractor requires transformers and torch. "
                    "Install them with: poetry install -E hf"
                ) from exc
            kwargs = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            self._model = AutoModel.from_pretrained(self.model_id, **kwargs)
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
