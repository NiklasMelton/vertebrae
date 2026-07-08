"""Shared helpers for optional extractor adapters."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.utils.validation import ensure_numeric_matrix

_IMAGE_MODES = {"auto", "grayscale", "preserve", "rgb"}
_ALPHA_MODES = {"black_background", "drop", "white_background"}


def callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"


def iter_chunks(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return value


def to_numpy_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_numpy_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_numpy_nested(item) for item in value)
    if isinstance(value, list):
        return [to_numpy_nested(item) for item in value]
    return tensor_to_numpy(value)


def maybe_move_to_device(value: Any, device: Optional[str], torch_module: Any) -> Any:
    if device is None:
        return value
    if isinstance(value, dict):
        return {
            key: maybe_move_to_device(item, device=device, torch_module=torch_module)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            maybe_move_to_device(item, device=device, torch_module=torch_module) for item in value
        )
    if isinstance(value, list):
        return [
            maybe_move_to_device(item, device=device, torch_module=torch_module) for item in value
        ]
    tensor_type = getattr(torch_module, "Tensor", None)
    if tensor_type is not None and isinstance(value, tensor_type):
        moved = value.to(device)
        return value if moved is None else moved
    if hasattr(value, "to") and callable(value.to):
        moved = value.to(device)
        return value if moved is None else moved
    return value


def call_model(model: Any, batch: Any) -> Any:
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


def ensure_text_sequence(value: Any, owner: str) -> List[str]:
    if isinstance(value, str):
        raise ValueError(f"{owner} expects a sequence of strings, not a single string.")
    try:
        texts = list(value)
    except TypeError as exc:
        raise ValueError(f"{owner} expects a sequence of strings.") from exc
    if not all(isinstance(text, str) for text in texts):
        raise ValueError(f"{owner} expects every input item to be a string.")
    return texts


def coerce_image(
    value: Any,
    image_module: Any,
    image_mode: str = "auto",
    alpha_mode: str = "drop",
) -> Any:
    if image_mode not in _IMAGE_MODES:
        raise ValueError(f"image_mode must be one of: {', '.join(sorted(_IMAGE_MODES))}.")
    if alpha_mode not in _ALPHA_MODES:
        raise ValueError(f"alpha_mode must be one of: {', '.join(sorted(_ALPHA_MODES))}.")
    if isinstance(value, (str, Path)):
        image = image_module.open(value)
        return _convert_image_mode(image, image_mode=image_mode, alpha_mode=alpha_mode)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 2:
            image = image_module.fromarray(array.astype(np.uint8))
            return _convert_image_mode(image, image_mode=image_mode, alpha_mode=alpha_mode)
        if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
            if array.shape[-1] == 1:
                array = array[:, :, 0]
            image = image_module.fromarray(array.astype(np.uint8))
            return _convert_image_mode(image, image_mode=image_mode, alpha_mode=alpha_mode)
    if hasattr(value, "convert"):
        return _convert_image_mode(value, image_mode=image_mode, alpha_mode=alpha_mode)
    return value


def resolve_output_specs(
    outputs: Optional[Sequence[Dict[str, Any]]],
    default_name: str = "embeddings",
) -> List[EmbeddingOutputSpec]:
    if outputs is None:
        return [EmbeddingOutputSpec(default_name)]
    if not outputs:
        raise ValueError("outputs must not be empty.")
    specs: List[EmbeddingOutputSpec] = []
    seen = set()
    for output in outputs:
        name = str(output.get("name", "")).strip()
        if not name:
            raise ValueError("Output specs must include a name.")
        if name in seen:
            raise ValueError("Output names must be unique.")
        metadata = dict(output.get("metadata") or {})
        if "selector" in output:
            metadata["selector"] = str(output["selector"])
        if "source" in output:
            metadata["source"] = str(output["source"])
        metadata["flatten"] = bool(output.get("flatten", True))
        specs.append(
            EmbeddingOutputSpec(
                name=name,
                pooling=str(output["pooling"]) if output.get("pooling") is not None else None,
                hidden_layer=(
                    int(output["hidden_layer"]) if output.get("hidden_layer") is not None else None
                ),
                metadata=metadata,
            )
        )
        seen.add(name)
    return specs


def materialize_named_outputs(
    raw_output: Any,
    specs: Sequence[EmbeddingOutputSpec],
    owner: str,
    allow_sparse: bool = False,
) -> List[EmbeddingOutput]:
    outputs: List[EmbeddingOutput] = []
    for spec in specs:
        value = resolve_output_value(raw_output, spec.metadata.get("selector"))
        embeddings = materialize_output_matrix(
            value,
            f"{owner} output '{spec.name}'",
            flatten=bool(spec.metadata.get("flatten", True)),
            allow_sparse=allow_sparse,
        )
        outputs.append(
            EmbeddingOutput(
                name=spec.name,
                embeddings=embeddings,
                recipe=spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
        )
    return outputs


def resolve_output_value(value: Any, selector: Optional[str]) -> Any:
    if selector is None or selector == "":
        return value
    current = value
    for part in selector.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def materialize_output_matrix(
    value: Any,
    name: str,
    flatten: bool = True,
    allow_sparse: bool = False,
) -> Any:
    if value is None:
        raise ValueError(f"{name} could not be resolved from the model output.")
    value = to_numpy_nested(value)
    if _is_sparse(value):
        return ensure_numeric_matrix(value, name, allow_sparse=allow_sparse)
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2 and flatten:
        array = array.reshape(array.shape[0], -1)
    return ensure_numeric_matrix(array, name, allow_sparse=allow_sparse)


def spec_to_recipe(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    recipe = {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
    }
    for key in ("selector", "source"):
        if key in spec.metadata:
            recipe[key] = spec.metadata[key]
    return recipe


def stack_batch(values: Sequence[Any], torch_module: Any = None) -> Any:
    if not values:
        return np.empty((0, 0), dtype=float)
    first = values[0]
    tensor_type = getattr(torch_module, "Tensor", None) if torch_module is not None else None
    if tensor_type is not None and isinstance(first, tensor_type):
        return torch_module.stack(list(values))
    if hasattr(first, "shape"):
        return np.stack([np.asarray(tensor_to_numpy(value)) for value in values], axis=0)
    return list(values)


def _convert_image_mode(value: Any, image_mode: str, alpha_mode: str) -> Any:
    if not hasattr(value, "convert"):
        return value
    if image_mode == "preserve":
        return value
    if image_mode == "grayscale":
        return value.convert("L")
    if image_mode in {"auto", "rgb"}:
        return value.convert("RGB")
    return value


def _is_sparse(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))
