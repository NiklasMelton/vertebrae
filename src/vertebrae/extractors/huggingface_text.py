"""Optional Hugging Face text embedding extractor."""

from typing import Any, Dict, List, Optional, cast

import numpy as np

from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec


class HFTextExtractor:
    """Hugging Face text backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        pooling: Pooling mode: `"mean"`, `"cls"`, or `"last_token"`.
        hidden_layer: Optional hidden-state layer index to pool from. Defaults to
            the model's final output.
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
        hidden_layer: Optional[int] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_paths: Optional[List[str]] = None,
    ) -> None:
        if pooling not in {"mean", "cls", "last_token"}:
            raise ValueError("pooling must be one of: mean, cls, last_token.")
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self._output_specs = _resolve_output_specs(
            outputs=outputs,
            default_pooling=pooling,
            default_hidden_layer=hidden_layer,
        )
        self._structured_output_specs = _resolve_structured_output_specs(structured_outputs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.checkpoint_paths = tuple(checkpoint_paths or ())
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

        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "HFTextExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode text inputs into dense embeddings.

        Args:
            X: Sequence of strings.
            y: Optional labels.

        Returns:
            Dense float32 embedding matrix.
        """

        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        tokenizer, model, torch = self._load_model()
        texts = _validate_text_sequence(X, "HFTextExtractor")
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        model.eval()
        need_hidden_states = any(spec.hidden_layer is not None for spec in self._output_specs)
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
                model_output = model(
                    **encoded,
                    output_hidden_states=need_hidden_states,
                )
                for spec in self._output_specs:
                    hidden = self._select_hidden_state(model_output, spec.hidden_layer)
                    pooled = self._pool(
                        hidden,
                        encoded["attention_mask"],
                        torch,
                        cast(str, spec.pooling),
                    )
                    collected[spec.name].append(
                        pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
        outputs: List[EmbeddingOutput] = []
        for spec in self._output_specs:
            arrays = collected[spec.name]
            embeddings = (
                np.vstack(arrays).astype(np.float32, copy=False) if arrays else np.empty((0, 0))
            )
            outputs.append(
                EmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    recipe={
                        "pooling": spec.pooling,
                        "hidden_layer": spec.hidden_layer,
                    },
                    metadata={
                        "pooling": spec.pooling,
                        "hidden_layer": spec.hidden_layer,
                    },
                )
            )
        return outputs

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if not self._structured_output_specs:
            raise ValueError("HFTextExtractor was not configured with structured_outputs.")
        tokenizer, model, torch = self._load_model()
        texts = _validate_text_sequence(X, "HFTextExtractor")
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }
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
                model_output = model(**encoded, output_hidden_states=True)
                for spec in self._structured_output_specs:
                    hidden = self._select_hidden_state(model_output, spec.hidden_layer)
                    values = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                    mask = encoded["attention_mask"].detach().cpu().numpy()
                    include_special = bool(spec.metadata.get("include_special_tokens", False))
                    for index in range(values.shape[0]):
                        length = int(mask[index].sum())
                        tokens = values[index, :length]
                        if not include_special and tokens.shape[0] >= 2:
                            tokens = tokens[1:-1]
                        collected[spec.name].append(tokens)
        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=collected[spec.name],
                unit_type=spec.unit_type,
                recipe={"hidden_layer": spec.hidden_layer},
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face text recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        recipe: Dict[str, Any] = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "pooling": self.pooling,
            "hidden_layer": self.hidden_layer,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "model_kwargs": self.model_kwargs,
            "streaming_safe": self.streaming_safe,
        }
        if len(self._output_specs) > 1:
            recipe["outputs"] = [_spec_to_dict(spec) for spec in self._output_specs]
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                _structured_spec_to_dict(spec) for spec in self._structured_output_specs
            ]
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        """Return Torch profiling hooks without forcing model loading."""

        from vertebrae.profiling import TorchResourceProfileAdapter

        return TorchResourceProfileAdapter(
            self,
            self.checkpoint_paths,
            model_getter=lambda: self._model,
            device_resolver=self._device,
        )

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

    def _select_hidden_state(self, output: Any, hidden_layer: Optional[int]) -> Any:
        if hidden_layer is None:
            return output.last_hidden_state
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "hidden_layer was requested, but model output has no hidden_states. "
                "This model may not support output_hidden_states."
            )
        try:
            return hidden_states[hidden_layer]
        except IndexError as exc:
            raise ValueError(
                f"hidden_layer index {hidden_layer} is out of range for "
                f"{len(hidden_states)} hidden states."
            ) from exc

    def _pool(self, hidden: Any, mask: Any, torch: Any, pooling: str) -> Any:
        if pooling == "cls":
            return hidden[:, 0, :]
        if pooling == "last_token":
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


def _resolve_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
    default_pooling: str,
    default_hidden_layer: Optional[int],
) -> List[EmbeddingOutputSpec]:
    if outputs is None:
        return [
            EmbeddingOutputSpec(
                name="default",
                pooling=default_pooling,
                hidden_layer=default_hidden_layer,
            )
        ]
    specs = []
    for raw in outputs:
        if "name" not in raw:
            raise ValueError("HFTextExtractor output specs must include a name.")
        pooling = raw.get("pooling", default_pooling)
        if pooling not in {"mean", "cls", "last_token"}:
            raise ValueError(
                "HFTextExtractor output pooling must be one of: mean, cls, last_token."
            )
        specs.append(
            EmbeddingOutputSpec(
                name=str(raw["name"]),
                pooling=pooling,
                hidden_layer=raw.get("hidden_layer"),
                metadata={},
            )
        )
    _ensure_unique_names(specs)
    return specs


def _ensure_unique_names(specs: List[Any]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("HFTextExtractor output names must be unique.")


def _spec_to_dict(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


def _resolve_structured_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
) -> List[StructuredOutputSpec]:
    specs = []
    for raw in outputs or []:
        if "name" not in raw:
            raise ValueError("HFTextExtractor structured outputs must include a name.")
        specs.append(
            StructuredOutputSpec(
                name=str(raw["name"]),
                unit_type=str(raw.get("unit_type", "token")),
                hidden_layer=raw.get("hidden_layer"),
                metadata={
                    "include_special_tokens": bool(raw.get("include_special_tokens", False)),
                },
            )
        )
    _ensure_unique_names(specs)
    return specs


def _structured_spec_to_dict(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
