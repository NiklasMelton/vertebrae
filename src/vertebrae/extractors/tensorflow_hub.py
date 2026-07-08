"""Optional TensorFlow Hub extractor."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vertebrae.extractors._utils import (
    callable_name,
    materialize_named_outputs,
    resolve_output_specs,
    spec_to_recipe,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec


class TFHubExtractor:
    """Wrap a TensorFlow Hub module as a vertebrae extractor."""

    def __init__(
        self,
        name: str,
        handle: str,
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: int = 32,
        modality: str = "unknown",
        model_kwargs: Optional[Dict[str, Any]] = None,
        call_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.handle = handle
        self.input_fn = input_fn or np.asarray
        self.output_fn = output_fn
        self._output_specs = resolve_output_specs(outputs)
        self.batch_size = batch_size
        self.modality = modality
        self.model_kwargs = model_kwargs or {}
        self.call_kwargs = call_kwargs or {}
        self.extractor_type = "tensorflow_hub"
        self.streaming_safe = True
        self._hub: Any = None
        self._model: Any = None

    def fit(self, X: Any, y: Any = None) -> "TFHubExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "TFHubExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        model = self._load_model()
        raw_inputs = self.input_fn(X)
        if callable(model):
            raw_output = model(raw_inputs, **self.call_kwargs)
        else:
            raw_output = model.signatures["default"](raw_inputs)
        projected = self.output_fn(raw_output) if self.output_fn is not None else raw_output
        outputs = materialize_named_outputs(
            projected,
            self._output_specs,
            owner=f"TFHubExtractor '{self.name}'",
            allow_sparse=False,
        )
        return [
            EmbeddingOutput(
                name=output.name,
                embeddings=np.asarray(output.embeddings, dtype=np.float32),
                recipe=spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for output, spec in zip(outputs, self._output_specs)
        ]

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "handle": self.handle,
            "input_fn": callable_name(self.input_fn),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "outputs": [spec_to_recipe(spec) for spec in self._output_specs],
            "batch_size": self.batch_size,
            "model_kwargs": self.model_kwargs,
            "call_kwargs": self.call_kwargs,
            "streaming_safe": self.streaming_safe,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import tensorflow_hub as hub
            except ImportError as exc:
                raise ImportError(
                    "TFHubExtractor requires optional TensorFlow Hub dependencies. "
                    "Install with `poetry install -E tensorflow-hub`."
                ) from exc
            self._hub = hub
            self._model = hub.load(self.handle, **self.model_kwargs)
        return self._model
