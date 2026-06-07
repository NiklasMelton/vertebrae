"""Optional ONNX runtime feature extractor."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix


class ONNXExtractor:
    """Wrap a local ONNX model as a feature extractor.

    Args:
        name: User-facing extractor name.
        model_path: Path to a local ONNX model file.
        input_fn: Optional callable that converts raw inputs into ONNX inputs.
        output_fn: Optional callable that converts raw ONNX outputs into embeddings.
        input_names: Optional ONNX input names to feed. Defaults to the model inputs.
        output_names: Optional ONNX output names to fetch. Defaults to the model outputs.
        providers: Optional ONNX Runtime execution providers.
        provider_options: Optional provider-specific configuration dictionaries.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable metadata for reproducibility.
        allow_sparse: Whether sparse embedding outputs are allowed.
        streaming_safe: Whether independent batches can be embedded without full-context state.
    """

    def __init__(
        self,
        name: str,
        model_path: Union[str, Path],
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_onnx",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = False,
        streaming_safe: bool = True,
    ) -> None:
        self.name = name
        self.model_path = Path(model_path)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.input_names = input_names
        self.output_names = output_names
        self.providers = providers
        self.provider_options = provider_options
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self._session: Any = None
        self._ort: Any = None

    def fit(self, X: Any, y: Any = None) -> "ONNXExtractor":
        """No-op fit for frozen ONNX models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Run the ONNX model and validate the resulting embeddings."""

        session = self._load_session()
        inputs = self._prepare_inputs(session, X)
        output_names = self._resolve_output_names(session)
        raw_outputs = session.run(output_names, inputs)
        embeddings = self._select_output(raw_outputs)
        return ensure_numeric_matrix(
            embeddings,
            f"ONNXExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Run the ONNX model and return validated embeddings."""

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable ONNX extractor recipe."""

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_path": str(self.model_path),
            "input_fn": _callable_name(self.input_fn) if self.input_fn is not None else None,
            "output_fn": _callable_name(self.output_fn) if self.output_fn is not None else None,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "providers": self.providers,
            "provider_options": self.provider_options,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
        }

    def _load_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "ONNXExtractor requires optional ONNX Runtime support. "
                    "Install with `poetry install -E onnx` or `pip install \"vertebrae[onnx]\"`."
                ) from exc
            session_kwargs: Dict[str, Any] = {}
            if self.providers is not None:
                session_kwargs["providers"] = self.providers
            if self.provider_options is not None:
                session_kwargs["provider_options"] = self.provider_options
            self._ort = ort
            self._session = ort.InferenceSession(str(self.model_path), **session_kwargs)
        return self._session

    def _prepare_inputs(self, session: Any, X: Any) -> Dict[str, Any]:
        if self.input_fn is not None:
            prepared = self.input_fn(X)
            if isinstance(prepared, dict):
                inputs: Dict[str, Any] = {}
                for key, value in prepared.items():
                    inputs[str(key)] = self._coerce_input_value(value)
                return inputs
            input_names = self._resolve_input_names(session, allow_multiple=False)
            return {input_names[0]: self._coerce_input_value(prepared)}

        input_names = self._resolve_input_names(session)
        if len(input_names) != 1:
            raise ValueError(
                "ONNXExtractor requires an input_fn when the model has multiple inputs."
            )
        return {input_names[0]: self._coerce_input_value(X)}

    def _resolve_input_names(self, session: Any, allow_multiple: bool = True) -> List[str]:
        if self.input_names is not None:
            if not self.input_names:
                raise ValueError("input_names must not be empty.")
            if not allow_multiple and len(self.input_names) != 1:
                raise ValueError(
                    "ONNXExtractor requires exactly one input when input_fn returns a single "
                    "array-like value."
                )
            return list(self.input_names)
        names = [str(item.name) for item in session.get_inputs()]
        if not names:
            raise ValueError("The ONNX model exposes no inputs.")
        if not allow_multiple and len(names) != 1:
            raise ValueError(
                "ONNXExtractor requires exactly one input when no input_names are provided."
            )
        return names

    def _resolve_output_names(self, session: Any) -> Optional[List[str]]:
        if self.output_names is not None:
            if not self.output_names:
                raise ValueError("output_names must not be empty.")
            return list(self.output_names)
        names = [str(item.name) for item in session.get_outputs()]
        if not names:
            raise ValueError("The ONNX model exposes no outputs.")
        if self.output_fn is not None:
            return names
        if len(names) != 1:
            raise ValueError(
                "ONNXExtractor requires an output_fn when the model has multiple outputs."
            )
        return names

    def _select_output(self, raw_outputs: Sequence[Any]) -> Any:
        if self.output_fn is not None:
            return self.output_fn(raw_outputs)
        if len(raw_outputs) != 1:
            raise ValueError(
                "ONNXExtractor received multiple outputs but no output_fn was provided."
            )
        return raw_outputs[0]

    def _coerce_input_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
