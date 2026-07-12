"""Optional Hugging Face multi-modal embedding extractor."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

import numpy as np

from vertebrae.extractors._utils import (
    materialize_structured_parent_matrices,
    resolve_output_value,
    structured_spec_to_recipe,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec

_TEXT_MODALITIES = {"text"}
_IMAGE_MODALITIES = {"image"}
_KNOWN_MODALITIES = {
    "audio",
    "embeddings",
    "fused",
    "image",
    "tabular",
    "text",
    "time_series",
    "video",
}


class HFMultimodalExtractor:
    """Hugging Face multi-modal backbone extractor with named outputs."""

    def __init__(
        self,
        name: str,
        model_id: str,
        input_modalities: Dict[str, str],
        outputs: List[Dict[str, Any]],
        processor_id: Optional[str] = None,
        input_map: Optional[Dict[str, str]] = None,
        input_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not input_modalities:
            raise ValueError("input_modalities must not be empty.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.name = name
        self.model_id = model_id
        self.processor_id = processor_id or model_id
        self.input_modalities = {str(key): str(value) for key, value in input_modalities.items()}
        self.input_map = dict(input_map or _default_input_map(self.input_modalities))
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.batch_size = batch_size
        self.image_mode = image_mode
        self.alpha_mode = alpha_mode
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.processor_kwargs = processor_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.modality = "multimodal"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._output_specs = _resolve_multimodal_output_specs(outputs)
        self._structured_output_specs = _resolve_multimodal_structured_output_specs(
            structured_outputs
        )
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._image_module: Any = None

        input_fields = set(self.input_modalities)
        map_fields = set(self.input_map)
        if not map_fields.issubset(input_fields):
            raise ValueError(
                "input_map may only reference declared input_modalities fields; "
                f"got unexpected fields {sorted(map_fields - input_fields)}."
            )

    def fit(self, X: Any, y: Any = None) -> "HFMultimodalExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "HFMultimodalExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return cast(np.ndarray, outputs[0].embeddings)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        processor, model, torch, image_module = self._load_model()
        batches = _normalize_multimodal_samples(X, self.input_modalities)
        model.eval()
        outputs_by_name: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._output_specs
        }

        with torch.no_grad():
            for batch in _iter_chunks(batches, self.batch_size):
                encoded = self._prepare_batch(batch, processor, torch, image_module)
                model_output = model(**encoded)
                projected = (
                    self.output_fn(model_output) if self.output_fn is not None else model_output
                )
                for spec in self._output_specs:
                    value = _resolve_named_output(projected, spec, model_output=model_output)
                    matrix = _materialize_output_matrix(
                        value,
                        spec=spec,
                        torch_module=torch,
                        batch_size=len(batch),
                    )
                    outputs_by_name[spec.name].append(matrix)

        materialized: List[EmbeddingOutput] = []
        for spec in self._output_specs:
            arrays = outputs_by_name[spec.name]
            embeddings = (
                np.vstack(arrays).astype(np.float32, copy=False) if arrays else np.empty((0, 0))
            )
            materialized.append(
                EmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    recipe=_spec_to_recipe(spec),
                    metadata=_spec_to_metadata(spec),
                )
            )
        return materialized

    def encode_retrieval(self, X: Any, *, branch: str, modality: str) -> np.ndarray:
        """Encode one independent branch when the wrapped model exposes it explicitly."""
        spec = next((item for item in self._output_specs if item.name == branch), None)
        if spec is None:
            raise ValueError(f"Unknown retrieval branch {branch!r}.")
        source = str(spec.metadata.get("source"))
        if source not in {"image", "text"} or modality != source:
            raise ValueError(
                f"Retrieval branch {branch!r} requires independently encodable {source!r} inputs."
            )
        processor, model, torch, image_module = self._load_model()
        method = getattr(model, f"get_{source}_features", None)
        if not callable(method):
            raise ValueError(
                f"Model {self.model_id!r} does not expose get_{source}_features(); provide "
                "a CallableRetrievalExtractor or an explicit retrieval-capable adapter."
            )
        field = next(
            (key for key, value in self.input_modalities.items() if value == source), source
        )
        values = list(X)
        outputs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = [{field: value} for value in values[start : start + self.batch_size]]
                raw = _default_processor_inputs(
                    batch=batch,
                    input_modalities={field: source},
                    input_map={field: "images" if source == "image" else "text"},
                    image_module=image_module,
                    image_mode=self.image_mode,
                    alpha_mode=self.alpha_mode,
                )
                encoded = processor(return_tensors="pt", **raw, **self.processor_kwargs)
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                value = method(**encoded)
                outputs.append(_materialize_output_matrix(value, spec, torch, len(batch)))
        return np.vstack(outputs).astype(np.float32, copy=False)

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if not self._structured_output_specs:
            raise ValueError("HFMultimodalExtractor was not configured with structured_outputs.")
        processor, model, torch, image_module = self._load_model()
        batches = _normalize_multimodal_samples(X, self.input_modalities)
        model.eval()
        outputs_by_name: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }

        with torch.no_grad():
            for batch in _iter_chunks(batches, self.batch_size):
                encoded = self._prepare_batch(batch, processor, torch, image_module)
                model_output = model(**encoded)
                projected = (
                    self.output_fn(model_output) if self.output_fn is not None else model_output
                )
                for spec in self._structured_output_specs:
                    value = _resolve_structured_output(projected, spec, model_output=model_output)
                    parents = materialize_structured_parent_matrices(
                        value,
                        f"HFMultimodalExtractor structured output '{spec.name}'",
                        expected_parents=len(batch),
                    )
                    outputs_by_name[spec.name].extend(
                        np.asarray(parent, dtype=np.float32) for parent in parents
                    )

        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=outputs_by_name[spec.name],
                unit_type=spec.unit_type,
                recipe=structured_spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "input_modalities": dict(self.input_modalities),
            "input_map": dict(self.input_map),
            "input_fn": _callable_name(self.input_fn) if self.input_fn is not None else None,
            "output_fn": _callable_name(self.output_fn) if self.output_fn is not None else None,
            "batch_size": self.batch_size,
            "image_mode": self.image_mode,
            "alpha_mode": self.alpha_mode,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "processor_kwargs": self.processor_kwargs,
            "model_kwargs": self.model_kwargs,
            "streaming_safe": self.streaming_safe,
            "outputs": [_spec_to_recipe(spec) for spec in self._output_specs],
        }
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                _structured_spec_to_recipe(spec) for spec in self._structured_output_specs
            ]
        return recipe

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from PIL import Image
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "HFMultimodalExtractor requires optional Hugging Face multi-modal "
                    "dependencies. Install with the documented Hugging Face extra or Poetry group."
                ) from exc
            common_kwargs = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            common_kwargs = {
                key: value for key, value in common_kwargs.items() if value is not None
            }
            self._processor = AutoProcessor.from_pretrained(
                self.processor_id,
                **common_kwargs,
                **self.processor_kwargs,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **common_kwargs,
                **self.model_kwargs,
            )
            self._torch = torch
            self._image_module = Image
            self._model.to(self._device(torch))
        return self._processor, self._model, self._torch, self._image_module

    def _device(self, torch: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare_batch(
        self,
        batch: List[Dict[str, Any]],
        processor: Any,
        torch: Any,
        image_module: Any,
    ) -> Dict[str, Any]:
        if self.input_fn is not None:
            raw_inputs = self.input_fn(_batch_to_inputs(batch))
        else:
            raw_inputs = _default_processor_inputs(
                batch=batch,
                input_modalities=self.input_modalities,
                input_map=self.input_map,
                image_module=image_module,
                image_mode=self.image_mode,
                alpha_mode=self.alpha_mode,
            )
        encoded = processor(return_tensors="pt", **raw_inputs, **self.processor_kwargs)
        return {key: value.to(self._device(torch)) for key, value in encoded.items()}


def _resolve_multimodal_output_specs(outputs: List[Dict[str, Any]]) -> List[EmbeddingOutputSpec]:
    if not outputs:
        raise ValueError("HFMultimodalExtractor outputs must not be empty.")
    specs: List[EmbeddingOutputSpec] = []
    seen = set()
    for output in outputs:
        name = str(output.get("name", "")).strip()
        if not name:
            raise ValueError("HFMultimodalExtractor output specs must include a name.")
        if name in seen:
            raise ValueError("HFMultimodalExtractor output names must be unique.")
        source = str(output.get("source", "")).strip()
        if not source:
            raise ValueError(f"HFMultimodalExtractor output '{name}' must include a source.")
        if source not in _KNOWN_MODALITIES:
            raise ValueError(
                f"HFMultimodalExtractor output '{name}' has unsupported source '{source}'."
            )
        model_output = str(output.get("model_output", "")).strip()
        if not model_output:
            raise ValueError(
                f"HFMultimodalExtractor output '{name}' must include a model_output selector."
            )
        pooling = output.get("pooling")
        hidden_layer = output.get("hidden_layer")
        metadata = {
            "source": source,
            "model_output": model_output,
        }
        if "selector" in output:
            metadata["selector"] = output["selector"]
        specs.append(
            EmbeddingOutputSpec(
                name=name,
                pooling=str(pooling) if pooling is not None else None,
                hidden_layer=int(hidden_layer) if hidden_layer is not None else None,
                metadata=metadata,
            )
        )
        seen.add(name)
    return specs


def _resolve_multimodal_structured_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
) -> List[StructuredOutputSpec]:
    if outputs is None:
        return []
    if not outputs:
        raise ValueError("HFMultimodalExtractor structured_outputs must not be empty.")
    specs: List[StructuredOutputSpec] = []
    seen = set()
    for output in outputs:
        name = str(output.get("name", "")).strip()
        if not name:
            raise ValueError("HFMultimodalExtractor structured output specs must include a name.")
        if name in seen:
            raise ValueError("HFMultimodalExtractor structured output names must be unique.")
        unit_type = str(output.get("unit_type", "")).strip()
        if not unit_type:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{name}' must include a unit_type."
            )
        source = str(output.get("source", "")).strip()
        if not source:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{name}' must include a source."
            )
        if source not in _KNOWN_MODALITIES:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{name}' has unsupported source "
                f"'{source}'."
            )
        model_output = str(output.get("model_output", "")).strip()
        if not model_output:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{name}' must include a "
                "model_output selector."
            )
        metadata = {
            "source": source,
            "model_output": model_output,
        }
        if output.get("selector") is not None:
            metadata["selector"] = str(output["selector"])
        specs.append(
            StructuredOutputSpec(
                name=name,
                unit_type=unit_type,
                hidden_layer=(
                    int(output["hidden_layer"]) if output.get("hidden_layer") is not None else None
                ),
                metadata=metadata,
            )
        )
        seen.add(name)
    return specs


def _spec_to_recipe(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "source": spec.metadata.get("source"),
        "model_output": spec.metadata.get("model_output"),
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "selector": spec.metadata.get("selector"),
    }


def _spec_to_metadata(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "source": spec.metadata.get("source"),
        "model_output": spec.metadata.get("model_output"),
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
    }


def _structured_spec_to_recipe(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "source": spec.metadata.get("source"),
        "model_output": spec.metadata.get("model_output"),
        "hidden_layer": spec.hidden_layer,
        "selector": spec.metadata.get("selector"),
    }


def _default_input_map(input_modalities: Dict[str, str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for field_name, modality in input_modalities.items():
        if modality in _TEXT_MODALITIES:
            result[field_name] = "text"
        elif modality in _IMAGE_MODALITIES:
            result[field_name] = "images"
        else:
            result[field_name] = field_name
    return result


def _normalize_multimodal_samples(X: Any, input_modalities: Dict[str, str]) -> List[Dict[str, Any]]:
    if not isinstance(X, dict):
        raise ValueError("HFMultimodalExtractor expects dict inputs with one entry per modality.")
    expected_fields = set(input_modalities)
    actual_fields = set(X)
    if actual_fields != expected_fields:
        raise ValueError(
            "HFMultimodalExtractor inputs must match declared input_modalities; "
            f"got {sorted(actual_fields)} expected {sorted(expected_fields)}."
        )
    n_samples = _structured_length(X)
    samples: List[Dict[str, Any]] = []
    for index in range(n_samples):
        sample = {}
        for field_name in input_modalities:
            sample[field_name] = _sample_at_index(X[field_name], index)
        samples.append(sample)
    return samples


def _structured_length(X: Dict[str, Any]) -> int:
    lengths = {key: len(value) for key, value in X.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Structured multi-modal inputs must align in length; found {lengths}.")
    return unique_lengths.pop()


def _sample_at_index(value: Any, index: int) -> Any:
    if hasattr(value, "iloc"):
        return value.iloc[index]
    return value[index]


def _batch_to_inputs(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if not batch:
        return {}
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def _default_processor_inputs(
    batch: List[Dict[str, Any]],
    input_modalities: Dict[str, str],
    input_map: Dict[str, str],
    image_module: Any,
    image_mode: str,
    alpha_mode: str,
) -> Dict[str, Any]:
    collected: Dict[str, List[Any]] = {}
    for field_name, processor_key in input_map.items():
        modality = input_modalities[field_name]
        values = [sample[field_name] for sample in batch]
        if modality in _TEXT_MODALITIES:
            collected[processor_key] = [str(value) for value in values]
        elif modality in _IMAGE_MODALITIES:
            collected[processor_key] = [
                _coerce_image(value, image_module, image_mode, alpha_mode) for value in values
            ]
        else:
            collected[processor_key] = values
    return collected


def _resolve_named_output(projected: Any, spec: Any, model_output: Any) -> Any:
    if isinstance(projected, dict) and spec.name in projected:
        return projected[spec.name]
    value = _resolve_path(projected, cast(str, spec.metadata["model_output"]))
    if value is None and projected is not model_output:
        value = _resolve_path(model_output, cast(str, spec.metadata["model_output"]))
    if value is None:
        raise ValueError(
            f"HFMultimodalExtractor output '{spec.name}' could not resolve "
            f"model_output='{spec.metadata['model_output']}'."
        )
    return value


def _resolve_structured_output(
    projected: Any,
    spec: StructuredOutputSpec,
    model_output: Any,
) -> Any:
    value = _resolve_named_output(projected, spec, model_output=model_output)
    selector = spec.metadata.get("selector")
    if selector is not None:
        resolved = resolve_output_value(value, str(selector))
        if resolved is None:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{spec.name}' could not resolve "
                f"selector='{selector}'."
            )
        value = resolved
    if spec.hidden_layer is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"HFMultimodalExtractor structured output '{spec.name}' requested hidden_layer "
                "but model_output does not resolve to hidden states."
            )
        try:
            value = value[spec.hidden_layer]
        except IndexError as exc:
            raise ValueError(
                f"HFMultimodalExtractor structured output '{spec.name}' hidden_layer "
                f"{spec.hidden_layer} is out of range."
            ) from exc
    return value


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _materialize_output_matrix(
    value: Any,
    spec: EmbeddingOutputSpec,
    torch_module: Any,
    batch_size: int,
) -> np.ndarray:
    if spec.hidden_layer is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"HFMultimodalExtractor output '{spec.name}' requested hidden_layer but "
                "model_output does not resolve to hidden states."
            )
        try:
            value = value[spec.hidden_layer]
        except IndexError as exc:
            raise ValueError(
                f"HFMultimodalExtractor output '{spec.name}' hidden_layer "
                f"{spec.hidden_layer} is out of range."
            ) from exc

    if spec.pooling is not None:
        value = _apply_pooling(value, pooling=spec.pooling, torch_module=torch_module)

    array = _to_numpy(value)
    if array.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                f"HFMultimodalExtractor output '{spec.name}' returned a 1D vector for "
                f"batch_size={batch_size}."
            )
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2:
        raise ValueError(
            f"HFMultimodalExtractor output '{spec.name}' must resolve to a 2D embedding matrix; "
            f"got shape {array.shape}."
        )
    return array.astype(np.float32, copy=False)


def _apply_pooling(value: Any, pooling: str, torch_module: Any) -> Any:
    if pooling == "cls":
        return value[:, 0, :]
    if pooling == "mean":
        return value.mean(dim=1)
    if pooling == "last_token":
        indices = torch_module.arange(value.shape[0], device=value.device)
        return value[indices, value.shape[1] - 1, :]
    if pooling == "pooler":
        return value
    raise ValueError(f"Unsupported pooling mode: {pooling}.")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _coerce_image(value: Any, image_module: Any, image_mode: str, alpha_mode: str) -> Any:
    if isinstance(value, (str, Path)):
        return image_module.open(value).convert("RGB")
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return image_module.fromarray(value.astype(np.uint8))
        if value.ndim == 3 and value.shape[-1] in {1, 3, 4}:
            if value.shape[-1] == 1:
                value = value[:, :, 0]
            elif value.shape[-1] == 4 and alpha_mode == "drop":
                value = value[:, :, :3]
            return image_module.fromarray(value.astype(np.uint8))
    return value


def _iter_chunks(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
