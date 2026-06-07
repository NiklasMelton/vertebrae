"""Optional Keras module extractor for local user-supplied models."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix


class KerasExtractor:
    """Wrap a locally loaded Keras model as a feature extractor.

    Args:
        name: User-facing extractor name.
        model: A locally loaded Keras model or compatible callable object.
        collate_fn: Callable that converts raw inputs into model inputs.
        output_fn: Optional callable that converts raw model output into embeddings.
        call_method: Whether to use ``model(...)`` or ``model.predict(...)``.
        call_kwargs: Extra keyword arguments passed when ``call_method="call"``.
        predict_kwargs: Extra keyword arguments passed when ``call_method="predict"``.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable metadata for reproducibility.
        allow_sparse: Whether sparse embedding outputs are accepted.
        streaming_safe: Whether independent batches can be embedded without full-context state.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        call_method: str = "call",
        call_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_keras",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = False,
        streaming_safe: bool = True,
    ) -> None:
        if call_method not in {"call", "predict"}:
            raise ValueError("call_method must be either 'call' or 'predict'.")
        self.name = name
        self.model = model
        self.collate_fn = collate_fn or np.asarray
        self.output_fn = output_fn
        self.call_method = call_method
        self.call_kwargs = {"training": False}
        if call_kwargs is not None:
            self.call_kwargs.update(call_kwargs)
        self.predict_kwargs = {"verbose": 0}
        if predict_kwargs is not None:
            self.predict_kwargs.update(predict_kwargs)
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self._keras: Any = None

    def fit(self, X: Any, y: Any = None) -> "KerasExtractor":
        """No-op fit for local Keras models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply the collate function, run the model, and validate embeddings."""

        self._load_keras()
        batch = self.collate_fn(X)
        model_output = self._call_model(batch)
        embeddings = self.output_fn(model_output) if self.output_fn is not None else model_output
        embeddings = self._to_numpy(embeddings)
        return ensure_numeric_matrix(
            embeddings,
            f"KerasExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the extractor and transform inputs."""

        return self.fit(X, y).transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable recipe for this extractor."""

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_class": self.model.__class__.__module__ + "." + self.model.__class__.__name__,
            "collate_fn": _callable_name(self.collate_fn),
            "output_fn": _callable_name(self.output_fn) if self.output_fn is not None else None,
            "call_method": self.call_method,
            "call_kwargs": self.call_kwargs,
            "predict_kwargs": self.predict_kwargs,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
        }

    def _load_keras(self) -> Any:
        if self._keras is None:
            try:
                import keras as keras_module
            except ImportError:
                try:
                    from tensorflow import keras as keras_module
                except ImportError as exc:
                    raise ImportError(
                        "KerasExtractor requires optional Keras support. Install with "
                        "`poetry install -E keras` or `poetry install -E tensorflow`."
                    ) from exc
            self._keras = keras_module
        return self._keras

    def _call_model(self, batch: Any) -> Any:
        if self.call_method == "predict":
            if not hasattr(self.model, "predict"):
                raise TypeError(
                    "KerasExtractor was configured with call_method='predict' but the model "
                    "does not expose predict()."
                )
            return self.model.predict(batch, **self.predict_kwargs)
        return self.model(batch, **self.call_kwargs)

    def _to_numpy(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._to_numpy(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_numpy(item) for item in value)
        if isinstance(value, list):
            return [self._to_numpy(item) for item in value]
        if hasattr(value, "numpy") and callable(value.numpy):
            return value.numpy()
        return value


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
