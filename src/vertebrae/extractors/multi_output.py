"""Generic multi-output extractor wrapper."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.utils.validation import ensure_numeric_matrix


class MultiOutputExtractor:
    """Wrap a callable that returns multiple named embedding matrices.

    Args:
        name: User-facing parent extractor name.
        output_specs: Named output descriptions.
        transform_many_fn: Callable returning a dict or sequence of embedding outputs.
        fit_fn: Optional callable invoked during ``fit``.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable data captured in ``recipe()``.
        allow_sparse: Whether sparse outputs are accepted.
        streaming_safe: Whether independent batches can be transformed safely.
    """

    def __init__(
        self,
        name: str,
        output_specs: List[EmbeddingOutputSpec],
        transform_many_fn: Callable[[Any], Any],
        fit_fn: Optional[Callable[[Any, Any], Any]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_multi_output",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = True,
        streaming_safe: bool = False,
    ) -> None:
        if not output_specs:
            raise ValueError("MultiOutputExtractor.output_specs must not be empty.")
        names = [spec.name for spec in output_specs]
        if len(set(names)) != len(names):
            raise ValueError("MultiOutputExtractor output names must be unique.")
        self.name = name
        self._output_specs = list(output_specs)
        self.transform_many_fn = transform_many_fn
        self.fit_fn = fit_fn
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe

    def fit(self, X: Any, y: Any = None) -> "MultiOutputExtractor":
        if self.fit_fn is not None:
            self.fit_fn(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "MultiOutputExtractor.transform() is only available when exactly one output "
                "is configured. Use Benchmark/Evaluator or transform_many()."
            )
        return ensure_numeric_matrix(
            outputs[0].embeddings,
            f"MultiOutputExtractor '{self.name}' output '{outputs[0].name}'",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        raw_outputs = self.transform_many_fn(X)
        if isinstance(raw_outputs, dict):
            items = list(raw_outputs.items())
            outputs = [
                EmbeddingOutput(name=str(name), embeddings=value, recipe={"output_name": str(name)})
                for name, value in items
            ]
        else:
            outputs = list(raw_outputs)
        output_by_name = {output.name: output for output in outputs}
        missing = [spec.name for spec in self._output_specs if spec.name not in output_by_name]
        if missing:
            raise ValueError(
                f"MultiOutputExtractor '{self.name}' did not return outputs for {missing}."
            )
        materialized: List[EmbeddingOutput] = []
        for spec in self._output_specs:
            output = output_by_name[spec.name]
            embeddings = ensure_numeric_matrix(
                output.embeddings,
                f"MultiOutputExtractor '{self.name}' output '{spec.name}'",
                allow_sparse=self.allow_sparse,
            )
            recipe = dict(output.recipe)
            recipe.setdefault("output_name", spec.name)
            materialized.append(
                EmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    recipe=recipe,
                    metadata=dict(output.metadata),
                )
            )
        return materialized

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "transform_many_fn": _callable_name(self.transform_many_fn),
            "fit_fn": _callable_name(self.fit_fn) if self.fit_fn is not None else None,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
            "outputs": [_spec_to_dict(spec) for spec in self._output_specs],
        }


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"


def _spec_to_dict(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
