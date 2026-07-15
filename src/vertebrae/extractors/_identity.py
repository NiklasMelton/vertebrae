"""Stable cache-identity helpers shared by extractor adapters."""

import builtins
import dis
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import marshal
import re
import struct
import sys
from datetime import date, datetime, time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple
from uuid import UUID

import numpy as np

from vertebrae.cache.fingerprint import exact_json_value

_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_CANONICAL_BUILTINS = dict(vars(builtins))
_UNSAFE_DYNAMIC_BUILTINS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "globals",
        "hash",
        "id",
        "input",
        "locals",
        "open",
        "repr",
        "vars",
    }
)


def validate_extractor_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name must be a non-empty string.")
    return value.strip()


def validate_cache_identity(value: Optional[str]) -> Optional[str]:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("cache_identity must be a non-empty string when provided.")
    return value.strip() if value is not None else None


def extractor_cache_reuse_decision(recipe: Any) -> Tuple[bool, str]:
    """Resolve canonical reuse eligibility and its user-facing status.

    ``cache_safe`` describes whether the declared identity is authoritative;
    ``cache_embeddings`` is an explicit opt-out even when that identity is safe.
    Distributed workflows still persist opt-out results for worker handoff, but
    must place them beneath a run-scoped key rather than treating them as cache
    entries.
    """

    if not isinstance(recipe, dict):
        raise TypeError("Extractor recipes must be dictionaries.")
    if recipe.get("cache_embeddings") is False:
        return False, "disabled"
    if recipe.get("cache_safe") is False:
        return False, "bypassed_unsafe_identity"
    return True, "miss"


def derived_cache_reuse_decision(
    *source_metadata: Any,
    identity_safe: bool = True,
) -> Tuple[bool, str]:
    """Propagate source cache eligibility through a derived artifact."""

    for metadata in source_metadata:
        if not isinstance(metadata, dict):
            raise TypeError("Derived cache source metadata must be dictionaries.")
        if metadata.get("cache_eligible", True) is False:
            status = metadata.get("cache_status")
            if status not in {"disabled", "bypassed_unsafe_identity"}:
                status = "disabled"
            return False, status
    if not identity_safe:
        return False, "bypassed_unsafe_identity"
    return True, "miss"


def cache_identity_fields(
    *,
    explicit: Optional[str],
    callables: Iterable[Tuple[str, Optional[Callable[..., Any]]]] = (),
    paths: Sequence[Any] = (),
    state_required: bool = False,
    require_pinned_revision: bool = False,
    revision: Optional[str] = None,
    revision_identifiers: Sequence[Any] = (),
    paths_authoritative: bool = True,
) -> Dict[str, Any]:
    """Build recipe fields and conservatively decide whether cache reuse is safe."""

    explicit = validate_cache_identity(explicit)
    callable_identities = {
        name: portable_callable_identity(value) for name, value in callables if value is not None
    }
    path_identities = [path_content_identity(path) for path in paths]
    pinned_revision = (
        revision if isinstance(revision, str) and _PINNED_REVISION.fullmatch(revision) else None
    )
    if explicit is not None:
        return {
            "cache_identity": explicit,
            "cache_safe": True,
            "callable_identities": callable_identities,
            "path_identities": path_identities,
            "pinned_revision": pinned_revision,
        }

    paths_valid = all(identity is not None for identity in path_identities)
    has_state_path = bool(path_identities) and paths_valid and paths_authoritative
    callables_valid = all(value is not None for value in callable_identities.values())
    state_safe = not state_required or has_state_path
    revision_identifier_identities = [
        path_content_identity(identifier) for identifier in revision_identifiers
    ]
    revision_safe = (
        not require_pinned_revision
        or pinned_revision is not None
        or (
            all(identity is not None for identity in revision_identifier_identities)
            if revision_identifier_identities
            else has_state_path
        )
    )
    return {
        "cache_identity": None,
        "cache_safe": bool(callables_valid and paths_valid and state_safe and revision_safe),
        "callable_identities": callable_identities,
        "path_identities": path_identities,
        "pinned_revision": pinned_revision,
    }


def portable_callable_identity(value: Callable[..., Any]) -> Optional[Dict[str, Any]]:
    """Return an import-resolvable callable path plus a digest of its implementation."""

    return _portable_callable_identity(
        value,
        active_paths=set(),
        observed_values=set(),
    )


def importable_callable_path(value: Callable[..., Any]) -> Optional[str]:
    """Return an exact import path independently of cache-state safety."""

    if getattr(value, "__self__", None) is not None:
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not isinstance(qualname, str)
        or module == "__main__"
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        return None
    try:
        resolved: Any = importlib.import_module(module)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (AttributeError, ImportError, ValueError):
        return None
    expected = getattr(value, "__func__", value)
    actual = getattr(resolved, "__func__", resolved)
    return f"{module}:{qualname}" if actual is expected else None


def _portable_callable_identity(
    value: Callable[..., Any],
    *,
    active_paths: Set[str],
    observed_values: Set[int],
) -> Optional[Dict[str, Any]]:
    """Build an identity including every referenced global dependency."""

    # An import path identifies the method implementation, not state captured by
    # the bound instance/class. Such state needs an explicit cache identity.
    bound_owner = getattr(value, "__self__", None)
    if bound_owner is not None and not isinstance(bound_owner, ModuleType):
        return None
    # A class body is not a complete identity for the mutable attributes on the
    # live class object. Referenced classes therefore require an explicit cache
    # identity rather than being treated like stateless functions.
    if inspect.isclass(value):
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or module == "__main__":
        return None
    if "<locals>" in qualname or "<lambda>" in qualname:
        return None
    path = f"{module}:{qualname}"
    if path in active_paths:
        # A function reachable through itself can inspect mutable runtime state
        # such as ``__globals__`` dynamically. A recursive marker cannot prove
        # that all such state has been captured, so require explicit opt-in.
        return None
    try:
        resolved: Any = importlib.import_module(module)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (AttributeError, ImportError, ValueError):
        return None
    expected = getattr(value, "__func__", value)
    actual = getattr(resolved, "__func__", resolved)
    if actual is not expected:
        return None

    implementation = _callable_implementation_bytes(expected)
    if implementation is None:
        return None
    if _contains_local_import(expected):
        # IMPORT_NAME resolves mutable live module state which cannot be proven by
        # the function's source/code digest alone.
        return None
    positional_defaults = _portable_state_identity(
        getattr(expected, "__defaults__", None),
        observed_values=observed_values,
    )
    keyword_defaults = _portable_state_identity(
        getattr(expected, "__kwdefaults__", None),
        observed_values=observed_values,
    )
    if positional_defaults is None or keyword_defaults is None:
        return None
    attributes = _portable_state_identity(
        getattr(expected, "__dict__", None),
        observed_values=observed_values,
    )
    annotations = _portable_state_identity(
        getattr(expected, "__annotations__", None),
        observed_values=observed_values,
    )
    documentation = _portable_state_identity(
        getattr(expected, "__doc__", None),
        observed_values=observed_values,
    )
    if attributes is None or annotations is None or documentation is None:
        return None
    globals_identity: Dict[str, Any] = {}
    builtins_identity: Dict[str, Any] = {}
    if inspect.isfunction(expected):
        try:
            closure = inspect.getclosurevars(expected)
        except TypeError:
            return None
        if closure.nonlocals or getattr(expected, "__closure__", None):
            return None
        if _contains_unsafe_identity_operation(expected):
            return None
        nested_active = {*active_paths, path}
        for name, dependency in sorted(closure.builtins.items()):
            dependency_identity = _builtin_dependency_identity(name, dependency)
            if dependency_identity is None:
                return None
            builtins_identity[name] = dependency_identity
        for name, dependency in sorted(closure.globals.items()):
            attribute_paths = (
                _module_attribute_paths(expected, name)
                if isinstance(dependency, ModuleType)
                else None
            )
            dependency_identity = _global_dependency_identity(
                dependency,
                active_paths=nested_active,
                observed_values=observed_values,
                module_attribute_paths=attribute_paths,
            )
            if dependency_identity is None:
                return None
            globals_identity[name] = dependency_identity
    return {
        "path": path,
        "sha256": hashlib.sha256(implementation).hexdigest(),
        "defaults": {
            "positional": positional_defaults,
            "keyword": keyword_defaults,
        },
        "attributes": attributes,
        "annotations": annotations,
        "documentation": documentation,
        "runtime_name": getattr(expected, "__name__", None),
        "runtime_qualname": getattr(expected, "__qualname__", None),
        "runtime_module": getattr(expected, "__module__", None),
        "builtins": builtins_identity,
        "globals": globals_identity,
    }


def _global_dependency_identity(
    value: Any,
    *,
    active_paths: Set[str],
    observed_values: Set[int],
    module_attribute_paths: Optional[Tuple[Tuple[str, ...], ...]] = None,
) -> Optional[Dict[str, Any]]:
    """Identify one global referenced by an importable callable."""

    if isinstance(value, ModuleType):
        return _module_identity(
            value,
            attribute_paths=module_attribute_paths,
            active_paths=active_paths,
            observed_values=observed_values,
        )
    if inspect.isclass(value):
        return None
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        identity = _portable_callable_identity(
            value,
            active_paths=active_paths,
            observed_values=observed_values,
        )
        return None if identity is None else {"type": "callable", "identity": identity}
    identity = _portable_state_identity(value, observed_values=observed_values)
    if identity is None:
        # Opaque live objects can carry mutable model state that is not represented
        # by their class or repr. In particular, do not infer safety from duck-typed
        # ``to_numpy`` or ``shape``/``dtype`` attributes.
        return None
    return {"type": "value", "value": identity}


def _portable_state_identity(
    value: Any,
    *,
    observed_values: Set[int],
) -> Optional[Any]:
    """Encode deterministic callable state without coercing opaque live objects.

    This intentionally accepts a smaller set than ``exact_json_value``. Callable
    defaults and globals participate in executable behavior, so a merely
    array-like object cannot be reduced to its current values while ignoring its
    mutable methods or other attributes. Local paths are content-addressed because
    their bytes, rather than the spelling of the path, affect callable behavior.
    """

    if isinstance(value, Path):
        path_identity = path_content_identity(value)
        return (
            None if path_identity is None else {"type": "path_content", "identity": path_identity}
        )
    if type(value) is float:
        return {"type": "float64_bits", "hex": struct.pack(">d", value).hex()}
    if type(value) is Decimal:
        decimal_tuple = value.as_tuple()
        return {
            "type": "decimal_tuple",
            "sign": decimal_tuple.sign,
            "digits": list(decimal_tuple.digits),
            "exponent": str(decimal_tuple.exponent),
        }
    if type(value) in {datetime, time} and value.tzinfo is not None:
        # isoformat() preserves the current offset but not the concrete tzinfo
        # implementation or its additional state (for example ZoneInfo.key).
        return None
    if value is None or type(value) in {
        bool,
        int,
        str,
        bytes,
        UUID,
        datetime,
        date,
        time,
        Fraction,
    }:
        return exact_json_value(value)
    if isinstance(value, np.generic):
        dtype_identity = _numpy_dtype_identity(value.dtype)
        if type(value).__module__ != "numpy" or dtype_identity is None:
            return None
        return {
            "type": "numpy_scalar",
            "class": f"{type(value).__module__}:{type(value).__qualname__}",
            "dtype": dtype_identity,
            "bytes": value.tobytes().hex(),
        }
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray or value.base is not None:
            return None
        dtype_identity = _numpy_dtype_identity(value.dtype)
        if dtype_identity is None:
            return None
        value_id = id(value)
        if value_id in observed_values:
            return None
        observed_values.add(value_id)
        if not value.dtype.hasobject:
            return {
                "value": exact_json_value(value),
                "dtype_identity": dtype_identity,
                "strides": list(value.strides),
                "writeable": bool(value.flags.writeable),
                "aligned": bool(value.flags.aligned),
                "c_contiguous": bool(value.flags.c_contiguous),
                "f_contiguous": bool(value.flags.f_contiguous),
            }
        return _portable_object_array_identity(
            value,
            observed_values=observed_values,
        )
    if _is_sparse_matrix(value):
        # Sparse implementations expose additional mutable structural flags and
        # subclass state beyond the component arrays represented by exact_json_value.
        return None

    value_type = type(value)
    if value_type in {set, frozenset}:
        # Hash-table iteration order is observable and is not determined solely
        # by sorted semantic members or insertion order across processes.
        return None
    if value_type not in {dict, list, tuple}:
        return None
    value_id = id(value)
    if value_id in observed_values:
        return None
    observed_values.add(value_id)
    if value_type is dict:
        items = []
        # Preserve insertion order. Runtime code may iterate over a dict, so
        # sorting its entries would collapse behaviorally distinct defaults.
        for key, item in value.items():
            key_identity = _portable_state_identity(
                key,
                observed_values=observed_values,
            )
            item_identity = _portable_state_identity(
                item,
                observed_values=observed_values,
            )
            if key_identity is None or item_identity is None:
                return None
            items.append([key_identity, item_identity])
        return {"type": "dict", "items": items}
    if value_type in {list, tuple}:
        items = []
        for item in value:
            item_identity = _portable_state_identity(
                item,
                observed_values=observed_values,
            )
            if item_identity is None:
                return None
            items.append(item_identity)
        return {
            "type": "list" if value_type is list else "tuple",
            "items": items,
        }
    return None


def _portable_object_array_identity(
    value: np.ndarray,
    *,
    observed_values: Set[int],
) -> Optional[Dict[str, Any]]:
    """Hash every object-array element after conservative state validation."""

    digest = hashlib.sha256()
    for index in np.ndindex(value.shape):
        item_identity = _portable_state_identity(
            value[index],
            observed_values=observed_values,
        )
        if item_identity is None:
            return None
        payload = _portable_json(item_identity).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return {
        "type": "object_ndarray",
        "shape": list(value.shape),
        "dtype_identity": _numpy_dtype_identity(value.dtype),
        "strides": list(value.strides),
        "writeable": bool(value.flags.writeable),
        "aligned": bool(value.flags.aligned),
        "c_contiguous": bool(value.flags.c_contiguous),
        "f_contiguous": bool(value.flags.f_contiguous),
        "values_sha256": digest.hexdigest(),
    }


def _portable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _numpy_dtype_identity(dtype: np.dtype[Any]) -> Optional[Dict[str, Any]]:
    """Encode dtype geometry and reject arbitrary attached metadata."""

    if dtype.hasobject or dtype.metadata is not None:
        return None
    return {
        "str": dtype.str,
        "descr": exact_json_value(dtype.descr),
        "itemsize": dtype.itemsize,
        "alignment": dtype.alignment,
        "byteorder": dtype.byteorder,
        "isalignedstruct": dtype.isalignedstruct,
    }


def _is_sparse_matrix(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))


def _module_attribute_paths(
    function: Callable[..., Any],
    global_name: str,
) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """Return statically resolved attribute chains for one module global.

    Passing a module object itself, or resolving attributes dynamically, cannot
    be covered by installed-file provenance because live module state is mutable.
    Such usage is deliberately marked unsafe.
    """

    try:
        instructions = tuple(dis.get_instructions(function))
    except (TypeError, ValueError):
        return None
    paths = set()
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_GLOBAL" or instruction.argval != global_name:
            continue
        path = []
        cursor = index + 1
        while cursor < len(instructions) and instructions[cursor].opname in {
            "LOAD_ATTR",
            "LOAD_METHOD",
        }:
            attribute = instructions[cursor].argval
            if not isinstance(attribute, str):
                return None
            path.append(attribute)
            cursor += 1
        if not path:
            return None
        paths.add(tuple(path))
    return tuple(sorted(paths)) if paths else None


def _builtin_dependency_identity(name: str, value: Any) -> Optional[Dict[str, Any]]:
    """Identify a canonical interpreter builtin and reject dynamic introspection."""

    if name in _UNSAFE_DYNAMIC_BUILTINS:
        return None
    marker = object()
    canonical = _CANONICAL_BUILTINS.get(name, marker)
    if canonical is marker or value is not canonical:
        return None
    if getattr(builtins, name, marker) is not canonical:
        return None
    return {
        "type": "canonical_builtin",
        "name": name,
        "object_type": f"{type(value).__module__}:{type(value).__qualname__}",
        "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_version": list(sys.version_info[:3]),
    }


def _contains_local_import(function: Callable[..., Any]) -> bool:
    if getattr(function, "__code__", None) is None:
        return False
    try:
        return any(
            instruction.opname in {"IMPORT_NAME", "IMPORT_FROM"}
            for instruction in dis.get_instructions(function)
        )
    except (TypeError, ValueError):
        return True


def _contains_unsafe_identity_operation(function: Callable[..., Any]) -> bool:
    """Reject identity comparisons whose operands are not direct inputs/constants."""

    try:
        instructions = tuple(dis.get_instructions(function))
    except (TypeError, ValueError):
        return True
    direct_operands = {"LOAD_CONST", "LOAD_FAST", "LOAD_DEREF"}
    for index, instruction in enumerate(instructions):
        if instruction.opname != "IS_OP":
            continue
        if index < 2 or any(
            operand.opname not in direct_operands for operand in instructions[index - 2 : index]
        ):
            return True
    return False


def _module_identity(
    module: ModuleType,
    *,
    attribute_paths: Optional[Tuple[Tuple[str, ...], ...]],
    active_paths: Set[str],
    observed_values: Set[int],
) -> Optional[Dict[str, Any]]:
    """Return module provenance plus the live values actually referenced."""

    if not attribute_paths:
        return None

    name = getattr(module, "__name__", None)
    identity: Dict[str, Any] = {"type": "module", "name": name}
    module_file = getattr(module, "__file__", None)
    file_identity = path_content_identity(module_file) if module_file else None
    if file_identity is not None:
        identity["file_sha256"] = file_identity["sha256"]
    if isinstance(name, str):
        package = name.split(".", 1)[0]
        package_map = getattr(importlib.metadata, "packages_distributions", None)
        distributions = package_map().get(package, ()) if package_map is not None else ()
        versions: Dict[str, str] = {}
        for distribution in sorted(distributions):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        declared_version = getattr(module, "__version__", None)
        if isinstance(declared_version, (str, int, float)):
            identity["declared_version"] = str(declared_version)
        if not versions:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        if versions:
            identity["distribution_versions"] = versions
    attributes = []
    for attribute_path in attribute_paths:
        current: Any = module
        try:
            for attribute in attribute_path:
                current = getattr(current, attribute)
        except (AttributeError, TypeError):
            return None
        current_identity = _global_dependency_identity(
            current,
            active_paths=active_paths,
            observed_values=observed_values,
        )
        if current_identity is None:
            return None
        attributes.append(
            {
                "path": list(attribute_path),
                "identity": current_identity,
            }
        )
    identity["referenced_attributes"] = attributes
    return identity


def path_content_identity(value: Any) -> Optional[Dict[str, Any]]:
    """Hash a local model/checkpoint file or directory by content."""

    try:
        unresolved = Path(value).expanduser()
        if unresolved.is_symlink():
            return None
        path = unresolved.resolve(strict=True)
    except (OSError, TypeError, ValueError):
        return None
    digest = hashlib.sha256()
    if path.is_file():
        _update_file_digest(digest, path)
        return {"path": str(path), "kind": "file", "sha256": digest.hexdigest()}
    if not path.is_dir():
        return None
    digest.update(b"vertebrae-directory-content/v2")
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        if child.is_symlink():
            return None
        kind = b"file" if child.is_file() else b"directory"
        content_digest = b""
        content_size = b""
        if child.is_file():
            file_digest = hashlib.sha256()
            _update_file_digest(file_digest, child)
            content_digest = file_digest.digest()
            content_size = child.stat().st_size.to_bytes(16, "big", signed=False)
        for field in (kind, relative, content_size, content_digest):
            digest.update(len(field).to_bytes(8, "big", signed=False))
            digest.update(field)
    return {"path": str(path), "kind": "directory", "sha256": digest.hexdigest()}


def local_model_paths(
    identifier: Any,
    extra_paths: Sequence[Any] = (),
    *,
    additional_identifiers: Sequence[Any] = (),
) -> Tuple[Any, ...]:
    """Return checkpoint paths plus every configured local component identifier."""

    paths = []
    observed = set()
    for candidate in (identifier, *additional_identifiers, *extra_paths):
        try:
            path = Path(candidate).expanduser()
            key = str(path)
            if key not in observed and path.exists():
                paths.append(candidate)
                observed.add(key)
        except (OSError, TypeError, ValueError):
            # Explicit checkpoint paths are retained so identity generation marks
            # missing or otherwise unreadable content as unsafe.
            if candidate in extra_paths:
                paths.append(candidate)
    return tuple(paths)


def _callable_implementation_bytes(value: Any) -> Optional[bytes]:
    runtime_code = getattr(value, "__code__", None)
    runtime_payload = marshal.dumps(runtime_code) if runtime_code is not None else b""
    try:
        source_payload = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        source_file: Optional[str]
        try:
            source_file = inspect.getsourcefile(value) or inspect.getfile(value)
        except (OSError, TypeError):
            module_name = getattr(value, "__module__", None)
            if not isinstance(module_name, str):
                source_file = None
            else:
                try:
                    source_file = getattr(importlib.import_module(module_name), "__file__", None)
                except (ImportError, TypeError, ValueError):
                    source_file = None
        identity = path_content_identity(source_file) if source_file else None
        if identity is None:
            return None
        source_payload = str(identity["sha256"]).encode("ascii")
    return b"".join(
        (
            len(source_payload).to_bytes(8, byteorder="big", signed=False),
            source_payload,
            len(runtime_payload).to_bytes(8, byteorder="big", signed=False),
            runtime_payload,
        )
    )


def _update_file_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
