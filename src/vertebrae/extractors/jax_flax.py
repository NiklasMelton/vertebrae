"""Optional JAX/Flax extractor."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vertebrae.extractors._utils import (
    callable_name,
    materialize_named_outputs,
    resolve_output_specs,
    spec_to_recipe,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec


class JAXFlaxExtractor:
    """Wrap a JAX/Flax apply function or model object as a vertebrae extractor."""

    def __init__(
        self,
        name: str,
        input_fn: Callable[[Any], Any],
        output_fn: Optional[Callable[[Any], Any]] = None,
        apply_fn: Optional[Callable[..., Any]] = None,
        model: Any = None,
        params: Any = None,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        modality: str = "unknown",
        jit: bool = True,
        apply_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if apply_fn is None and model is None:
            raise ValueError("JAXFlaxExtractor requires either apply_fn or model.")
        self.name = name
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.apply_fn = apply_fn
        self.model = model
        self.params = params
        self._output_specs = resolve_output_specs(outputs)
        self.modality = modality
        self.jit = jit
        self.apply_kwargs = apply_kwargs or {}
        self.extractor_type = "jax_flax"
        self.streaming_safe = True
        self._jax: Any = None
        self._compiled_apply: Optional[Callable[..., Any]] = None

    def fit(self, X: Any, y: Any = None) -> "JAXFlaxExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "JAXFlaxExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        raw_inputs = self.input_fn(X)
        raw_output = self._apply(raw_inputs)
        projected = self.output_fn(raw_output) if self.output_fn is not None else raw_output
        outputs = materialize_named_outputs(
            projected,
            self._output_specs,
            owner=f"JAXFlaxExtractor '{self.name}'",
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
            "input_fn": callable_name(self.input_fn),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "apply_fn": callable_name(self.apply_fn) if self.apply_fn is not None else None,
            "model_class": (
                self.model.__class__.__module__ + "." + self.model.__class__.__name__
                if self.model is not None
                else None
            ),
            "has_params": self.params is not None,
            "outputs": [spec_to_recipe(spec) for spec in self._output_specs],
            "jit": self.jit,
            "apply_kwargs": self.apply_kwargs,
            "streaming_safe": self.streaming_safe,
        }

    def _apply(self, inputs: Any) -> Any:
        self._load_jax()
        compiled = self._compiled_apply
        if compiled is None:
            fn = self._apply_impl
            compiled = self._jax.jit(fn) if self.jit else fn
            self._compiled_apply = compiled
        return compiled(inputs)

    def _apply_impl(self, inputs: Any) -> Any:
        if self.apply_fn is not None:
            if self.params is not None:
                return self.apply_fn(self.params, inputs, **self.apply_kwargs)
            return self.apply_fn(inputs, **self.apply_kwargs)
        assert self.model is not None
        if hasattr(self.model, "apply"):
            variables = self.params if self.params is not None else {}
            return self.model.apply(variables, inputs, **self.apply_kwargs)
        if self.params is not None:
            return self.model(self.params, inputs, **self.apply_kwargs)
        return self.model(inputs, **self.apply_kwargs)

    def _load_jax(self) -> None:
        if self._jax is None:
            try:
                import flax  # noqa: F401
                import jax
            except ImportError as exc:
                raise ImportError(
                    "JAXFlaxExtractor requires optional JAX/Flax dependencies. "
                    "Install with `poetry install -E jax`."
                ) from exc
            self._jax = jax
