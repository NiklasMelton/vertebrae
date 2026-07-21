"""Shared helpers for optional extractor adapters."""

from copy import deepcopy
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from vertebrae.extractors._outputs import validate_named_output_mapping
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix

_IMAGE_MODES = {"auto", "grayscale", "preserve", "rgb"}
_ALPHA_MODES = {"black_background", "drop", "white_background"}


def optional_dependency_versions(*distributions: str) -> Dict[str, Optional[str]]:
    """Resolve optional distribution versions without importing their runtime modules.

    Missing distributions remain explicit ``None`` entries. This keeps recipes
    deterministic in lightweight environments while ensuring that an installed
    backend upgrade changes the extractor fingerprint.
    """

    versions: Dict[str, Optional[str]] = {}
    for distribution in sorted(set(distributions)):
        normalized = validate_nonblank_string(distribution, "distribution name")
        try:
            versions[normalized] = importlib_metadata.version(normalized)
        except importlib_metadata.PackageNotFoundError:
            versions[normalized] = None
    return versions


def callable_name(fn: Callable[..., Any]) -> str:
    value_type = type(fn)
    module = getattr(fn, "__module__", value_type.__module__)
    qualname = getattr(fn, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"


def iter_chunks(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    validate_batch_size(size, "chunk size")
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def validate_batch_size(value: Any, owner: str = "batch_size") -> int:
    """Validate a public batch-size option before any iteration starts."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner} must be an integer.")
    if int(value) < 1:
        raise ValueError(f"{owner} must be >= 1.")
    return int(value)


def validate_nonblank_string(value: Any, owner: str) -> str:
    """Return a normalized, nonblank public string option."""

    if not isinstance(value, str):
        raise TypeError(f"{owner} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{owner} must be a non-empty string.")
    return normalized


def validate_optional_nonblank_string(value: Any, owner: str) -> Optional[str]:
    """Validate an optional string without treating blank input as absent."""

    if value is None:
        return None
    return validate_nonblank_string(value, owner)


def validate_bool(value: Any, owner: str) -> bool:
    """Reject truthy integer/string substitutes for public boolean options."""

    if not isinstance(value, bool):
        raise TypeError(f"{owner} must be a bool.")
    return value


def validate_choice(value: Any, owner: str, choices: Iterable[str]) -> str:
    """Validate a string enum and return it unchanged after whitespace checks."""

    normalized = validate_nonblank_string(value, owner)
    allowed = tuple(choices)
    if normalized not in allowed:
        rendered = ", ".join(repr(choice) for choice in sorted(allowed))
        raise ValueError(f"{owner} must be one of: {rendered}.")
    return normalized


def snapshot_mapping(
    value: Any,
    owner: str,
    *,
    require_string_keys: bool = True,
    serializable: bool = True,
) -> Dict[Any, Any]:
    """Validate and defensively copy constructor mappings.

    Optional extractor mappings become part of recipes/cache identities, so they
    must not change when a caller later mutates the object passed to the
    constructor. Keyword-argument mappings also need string keys because they are
    eventually expanded with ``**``.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping when provided.")
    if require_string_keys:
        for key in value:
            validate_nonblank_string(key, f"{owner} keys")
    try:
        snapshot = deepcopy(dict(value))
    except Exception as exc:
        raise TypeError(f"{owner} must be defensively copyable.") from exc
    if serializable:
        try:
            make_json_safe(snapshot)
        except TypeError as exc:
            raise TypeError(f"{owner} must contain serializable values: {exc}") from exc
    return snapshot


def snapshot_string_mapping(
    value: Any,
    owner: str,
    *,
    allowed_values: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Snapshot a mapping whose keys and values are nonblank strings."""

    snapshot = snapshot_mapping(value, owner, serializable=True)
    allowed = set(allowed_values) if allowed_values is not None else None
    result: Dict[str, str] = {}
    for key, item in snapshot.items():
        normalized_key = validate_nonblank_string(key, f"{owner} keys")
        normalized_value = validate_nonblank_string(item, f"{owner}[{normalized_key!r}]")
        if allowed is not None and normalized_value not in allowed:
            rendered = ", ".join(repr(choice) for choice in sorted(allowed))
            raise ValueError(f"{owner}[{normalized_key!r}] must be one of: {rendered}.")
        result[normalized_key] = normalized_value
    return result


def snapshot_string_sequence(value: Any, owner: str) -> Optional[List[str]]:
    """Snapshot an optional sequence of nonblank strings."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} must be a sequence of strings, not a string.")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{owner} must be a sequence of strings.") from exc
    return [validate_nonblank_string(item, f"{owner}[{index}]") for index, item in enumerate(items)]


def snapshot_mapping_sequence(value: Any, owner: str) -> Optional[List[Dict[Any, Any]]]:
    """Snapshot an optional sequence of serializable mappings."""

    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{owner} must be a sequence of mappings.")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{owner} must be a sequence of mappings.") from exc
    return [snapshot_mapping(item, f"{owner}[{index}]") for index, item in enumerate(items)]


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
        return _convert_image_mode(
            image, image_mode=image_mode, alpha_mode=alpha_mode, image_module=image_module
        )
    if isinstance(value, np.ndarray):
        array = _image_array_to_uint8(value)
        if array.ndim == 2:
            image = image_module.fromarray(array)
            return _convert_image_mode(
                image, image_mode=image_mode, alpha_mode=alpha_mode, image_module=image_module
            )
        if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
            if array.shape[-1] == 1:
                array = array[:, :, 0]
            image = image_module.fromarray(array)
            return _convert_image_mode(
                image, image_mode=image_mode, alpha_mode=alpha_mode, image_module=image_module
            )
        raise ValueError(
            "NumPy image arrays must have shape (height, width), "
            "(height, width, 1), (height, width, 3), or (height, width, 4); "
            "the channel dimension must contain 1, 3, or 4 channels."
        )
    if hasattr(value, "convert"):
        return _convert_image_mode(
            value, image_mode=image_mode, alpha_mode=alpha_mode, image_module=image_module
        )
    return value


def resolve_output_specs(
    outputs: Optional[Sequence[Dict[str, Any]]],
    default_name: str = "embeddings",
    *,
    implicit_flatten: bool = True,
) -> List[EmbeddingOutputSpec]:
    if outputs is None:
        return [
            EmbeddingOutputSpec(
                default_name,
                metadata={"flatten": validate_bool(implicit_flatten, "implicit_flatten")},
            )
        ]
    if not outputs:
        raise ValueError("outputs must not be empty.")
    specs: List[EmbeddingOutputSpec] = []
    seen = set()
    for output in outputs:
        if not isinstance(output, Mapping):
            raise TypeError("Every output spec must be a mapping.")
        raw_name = output.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Output specs must include a name.")
        name = raw_name.strip()
        if name in seen:
            raise ValueError("Output names must be unique.")
        raw_metadata = output.get("metadata", {})
        metadata = snapshot_mapping(raw_metadata, "Output spec metadata")
        if "selector" in output:
            if not isinstance(output["selector"], str):
                raise TypeError("Output spec selector must be a string.")
            metadata["selector"] = output["selector"]
        if "source" in output:
            if not isinstance(output["source"], str):
                raise TypeError("Output spec source must be a string.")
            metadata["source"] = output["source"]
        flatten = output.get("flatten", True)
        if not isinstance(flatten, bool):
            raise TypeError("Output spec flatten must be a bool.")
        metadata["flatten"] = flatten
        pooling = output.get("pooling")
        if pooling is not None and not isinstance(pooling, str):
            raise TypeError("Output spec pooling must be a string when provided.")
        specs.append(
            EmbeddingOutputSpec(
                name=name,
                pooling=pooling,
                hidden_layer=_optional_integral(output.get("hidden_layer"), "hidden_layer"),
                metadata=metadata,
            )
        )
        seen.add(name)
    return specs


def validate_ordinary_output_projection(
    value: Any,
    specs: Sequence[EmbeddingOutputSpec],
    owner: str,
) -> Any:
    """Validate how selector-free specs obtain values from a multi-output projection."""

    if len(specs) <= 1:
        return value
    selector_free = [spec for spec in specs if not str(spec.metadata.get("selector") or "").strip()]
    if not selector_free:
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{owner} multi-output adapters with selector-free outputs must return a mapping."
        )
    if len(selector_free) == len(specs):
        return validate_named_output_mapping(
            value,
            [spec.name for spec in specs],
            f"{owner} ordinary output",
        )
    missing = [spec.name for spec in selector_free if spec.name not in value]
    if missing:
        raise ValueError(
            f"{owner} selector-free output names must be present in the projected mapping; "
            f"missing={missing}."
        )
    return dict(value)


def materialize_named_outputs(
    raw_output: Any,
    specs: Sequence[EmbeddingOutputSpec],
    owner: str,
    allow_sparse: bool = False,
    fallback_output: Any = None,
    allow_1d: bool = True,
) -> List[EmbeddingOutput]:
    outputs: List[EmbeddingOutput] = []
    for spec in specs:
        if (
            spec.metadata.get("selector") is None
            and isinstance(raw_output, Mapping)
            and spec.name in raw_output
        ):
            value = raw_output[spec.name]
        else:
            value = resolve_output_value(raw_output, spec.metadata.get("selector"))
        if value is None and fallback_output is not None and fallback_output is not raw_output:
            if (
                spec.metadata.get("selector") is None
                and isinstance(fallback_output, Mapping)
                and spec.name in fallback_output
            ):
                value = fallback_output[spec.name]
            else:
                value = resolve_output_value(fallback_output, spec.metadata.get("selector"))
        embeddings = materialize_output_matrix(
            value,
            f"{owner} output '{spec.name}'",
            flatten=bool(spec.metadata.get("flatten", True)),
            allow_sparse=allow_sparse,
            allow_1d=allow_1d,
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


def resolve_structured_output_specs(
    outputs: Optional[Sequence[Dict[str, Any]]],
) -> List[StructuredOutputSpec]:
    if outputs is None:
        return []
    if not outputs:
        raise ValueError("structured_outputs must not be empty.")
    specs: List[StructuredOutputSpec] = []
    seen = set()
    for output in outputs:
        if not isinstance(output, Mapping):
            raise TypeError("Every structured output spec must be a mapping.")
        raw_name = output.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Structured output specs must include a name.")
        name = raw_name.strip()
        if name in seen:
            raise ValueError("Structured output names must be unique.")
        raw_unit_type = output.get("unit_type")
        if not isinstance(raw_unit_type, str) or not raw_unit_type.strip():
            raise ValueError(f"Structured output '{name}' must include a unit_type.")
        unit_type = raw_unit_type.strip()
        raw_metadata = output.get("metadata", {})
        metadata = snapshot_mapping(raw_metadata, "Structured output spec metadata")
        for key in ("selector", "source", "model_output"):
            if output.get(key) is not None:
                if not isinstance(output[key], str):
                    raise TypeError(f"Structured output spec {key} must be a string.")
                metadata[key] = output[key]
        specs.append(
            StructuredOutputSpec(
                name=name,
                unit_type=unit_type,
                hidden_layer=_optional_integral(output.get("hidden_layer"), "hidden_layer"),
                metadata=metadata,
            )
        )
        seen.add(name)
    return specs


def _optional_integral(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Output spec {name} must be an integer when provided.")
    return int(value)


def materialize_named_structured_outputs(
    projected: Any,
    specs: Sequence[StructuredOutputSpec],
    owner: str,
    raw_output: Any = None,
    expected_parents: Optional[int] = None,
) -> List[StructuredEmbeddingOutput]:
    outputs: List[StructuredEmbeddingOutput] = []
    for spec in specs:
        if (
            spec.metadata.get("selector") is None
            and isinstance(projected, dict)
            and spec.name in projected
        ):
            value = projected[spec.name]
        else:
            value = resolve_output_value(projected, spec.metadata.get("selector"))
        if value is None and raw_output is not None and raw_output is not projected:
            if (
                spec.metadata.get("selector") is None
                and isinstance(raw_output, dict)
                and spec.name in raw_output
            ):
                value = raw_output[spec.name]
            else:
                value = resolve_output_value(raw_output, spec.metadata.get("selector"))
        embeddings = materialize_structured_parent_matrices(
            value,
            f"{owner} structured output '{spec.name}'",
            expected_parents=expected_parents,
        )
        outputs.append(
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=embeddings,
                unit_type=spec.unit_type,
                recipe=structured_spec_to_recipe(spec),
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
        if isinstance(current, Mapping):
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
    allow_1d: bool = True,
) -> Any:
    if value is None:
        raise ValueError(f"{name} could not be resolved from the model output.")
    value = to_numpy_nested(value)
    if _is_sparse(value):
        return ensure_numeric_matrix(value, name, allow_sparse=allow_sparse)
    array = np.asarray(value)
    if array.ndim == 1:
        if not allow_1d:
            raise ValueError(f"{name} must be a 2D numeric matrix.")
        array = array.reshape(1, -1)
    elif array.ndim > 2 and flatten:
        array = array.reshape(array.shape[0], -1)
    return ensure_numeric_matrix(array, name, allow_sparse=allow_sparse)


def materialize_structured_parent_matrices(
    value: Any,
    name: str,
    expected_parents: Optional[int] = None,
) -> List[Any]:
    if value is None:
        raise ValueError(f"{name} could not be resolved from the model output.")
    value = to_numpy_nested(value)
    if _is_sparse(value):
        parents = [ensure_numeric_matrix(value, name, allow_sparse=True)]
    elif isinstance(value, np.ndarray):
        if value.ndim == 3:
            parents = [
                ensure_numeric_matrix(value[index], name, allow_sparse=True)
                for index in range(value.shape[0])
            ]
        elif value.ndim == 2:
            parents = [ensure_numeric_matrix(value, name, allow_sparse=True)]
        elif value.ndim == 1 and value.dtype == object:
            parents = [
                ensure_numeric_matrix(item, name, allow_sparse=True) for item in value.tolist()
            ]
        else:
            raise ValueError(
                f"{name} must resolve to a batched 3D array or a sequence of per-parent "
                f"2D matrices; got shape {value.shape}."
            )
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must resolve to a batched 3D array or a sequence of per-parent 2D "
                "matrices."
            ) from exc
        parents = [ensure_numeric_matrix(item, name, allow_sparse=True) for item in items]
    if expected_parents is not None and len(parents) != expected_parents:
        raise ValueError(
            f"{name} returned {len(parents)} parents for a batch of {expected_parents}."
        )
    return parents


def spec_to_recipe(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    recipe = {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
    for key in ("selector", "source", "flatten"):
        if key in spec.metadata:
            recipe[key] = spec.metadata[key]
    return recipe


def structured_spec_to_recipe(spec: StructuredOutputSpec) -> Dict[str, Any]:
    recipe = {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
    for key in ("selector", "source", "model_output"):
        if key in spec.metadata:
            recipe[key] = spec.metadata[key]
    return recipe


def _image_array_to_uint8(value: np.ndarray) -> np.ndarray:
    """Convert supported image arrays without silent wrapping or float truncation."""

    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("Floating-point image arrays must contain only finite values.")
        if array.size and (float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0):
            raise ValueError("Floating-point image arrays must have values in the [0, 1] range.")
        return np.rint(array * 255.0).astype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        if array.size and (int(np.min(array)) < 0 or int(np.max(array)) > 255):
            raise ValueError("Integer image arrays must have values in the [0, 255] range.")
        return array.astype(np.uint8, copy=False)
    raise TypeError("NumPy image arrays must use an integer or floating-point dtype.")


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


def infer_batch_size(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 0
        return infer_batch_size(next(iter(value.values())))
    return int(len(value))


def _convert_image_mode(value: Any, image_mode: str, alpha_mode: str, image_module: Any) -> Any:
    if not hasattr(value, "convert"):
        return value
    if image_mode == "preserve":
        return value
    bands = tuple(value.getbands()) if hasattr(value, "getbands") else ()
    if "A" in bands:
        rgba = value.convert("RGBA")
        if alpha_mode == "drop":
            value = rgba.convert("RGB")
        else:
            fill = "white" if alpha_mode == "white_background" else "black"
            background = image_module.new("RGB", rgba.size, fill)
            background.paste(rgba, mask=rgba.getchannel("A"))
            value = background
    if image_mode == "grayscale":
        grayscale = np.asarray(value.convert("L"), dtype=np.uint8)
        return grayscale[:, :, np.newaxis]
    if image_mode in {"auto", "rgb"}:
        return value.convert("RGB")
    return value


def _is_sparse(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))
