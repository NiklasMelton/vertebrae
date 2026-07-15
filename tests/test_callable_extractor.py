import builtins
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from vertebrae import SpatialLayout, SpatialOutputSpec, StructuredOutputSpec
from vertebrae.extractors import (
    CallableExtractor,
    CallableSpatialExtractor,
    CallableStructuredExtractor,
    HFAudioExtractor,
    HFMultimodalExtractor,
    HFTextExtractor,
    HFTimeSeriesExtractor,
    HFVideoExtractor,
    HFVisionExtractor,
    PrecomputedSpatialExtractor,
    PrecomputedStructuredExtractor,
)
from vertebrae.extractors._identity import path_content_identity
from vertebrae.extractors.base import EmbeddingOutputSpec


def _portable_transform(value):
    return np.asarray(value, dtype=float)


_GLOBAL_SCALE = 1.0


def _global_config_transform(value):
    return np.asarray(value, dtype=float) * _GLOBAL_SCALE


class _StatefulTransform:
    def __init__(self, scale):
        self.scale = scale

    def transform(self, value):
        return np.asarray(value, dtype=float) * self.scale

    def __call__(self, value):
        return self.transform(value)


_GLOBAL_MODEL = _StatefulTransform(1.0)


def _global_model_transform(value):
    return _GLOBAL_MODEL.transform(value)


class _MutableGlobalConfig:
    scale = 1.0


def _global_class_transform(value):
    return np.asarray(value, dtype=float) * _MutableGlobalConfig.scale


def _function_attribute_dependency(value):
    return value


_function_attribute_dependency.scale = 1.0


def _function_attribute_transform(value):
    return np.asarray(value, dtype=float) * _function_attribute_dependency.scale


def _self_metadata_transform(value):
    scale = 1.0 + len(_self_metadata_transform.__annotations__)
    return np.asarray(value, dtype=float) * scale


class _ToNumpyState:
    def __init__(self, scale):
        self.scale = scale

    def to_numpy(self):
        return np.asarray([self.scale])


class _ArrayProtocolState:
    shape = (1,)
    dtype = np.dtype(float)

    def __init__(self, scale):
        self.scale = scale

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray([self.scale], dtype=dtype)


_GLOBAL_DUCK_STATE = _ToNumpyState(1.0)


def _duck_state_transform(value):
    return np.asarray(value, dtype=float) * np.asarray(_GLOBAL_DUCK_STATE)[0]


_GLOBAL_CHECKPOINT_PATH = Path(__file__)


def _global_path_transform(value):
    return np.asarray(value, dtype=float) * len(_GLOBAL_CHECKPOINT_PATH.read_bytes())


def _default_path_transform(value, checkpoint=Path(__file__)):
    return np.asarray(value, dtype=float) * len(checkpoint.read_bytes())


_MODULE_SCALE = 1.0
_CONFIG_MODULE = sys.modules[__name__]


def _module_attribute_transform(value):
    return np.asarray(value, dtype=float) * _CONFIG_MODULE._MODULE_SCALE


_GLOBAL_ALIAS_CONFIG = [[], []]


def _alias_sensitive_transform(value):
    scale = 1.0 if _GLOBAL_ALIAS_CONFIG[0] is _GLOBAL_ALIAS_CONFIG[1] else 2.0
    return np.asarray(value, dtype=float) * scale


_GLOBAL_ARRAY_STATE = np.array([[1.0, 2.0], [3.0, 4.0]], order="C")


def _array_layout_transform(value):
    scale = 1.0 if _GLOBAL_ARRAY_STATE.flags.c_contiguous else 2.0
    return np.asarray(value, dtype=float) * scale


def _runtime_code_transform(value):
    return np.asarray(value, dtype=float) * 1.0


def _replacement_runtime_code(value):
    return np.asarray(value, dtype=float) * 2.0


def _builtin_transform(value):
    return np.asarray(value, dtype=float) * abs(-3.0)


def _dynamic_global_transform(value):
    return np.asarray(value, dtype=float) * globals()["_GLOBAL_SCALE"]


def _local_import_transform(value):
    import math

    return np.asarray(value, dtype=float) * math.floor(1.9)


_GLOBAL_DECIMAL_SCALE = Decimal("1.0")


def _decimal_representation_transform(value):
    exponent = _GLOBAL_DECIMAL_SCALE.as_tuple().exponent
    return np.asarray(value, dtype=float) * float(exponent)


_GLOBAL_AWARE_DATETIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _aware_datetime_transform(value):
    scale = 1.0 if hasattr(_GLOBAL_AWARE_DATETIME.tzinfo, "key") else 2.0
    return np.asarray(value, dtype=float) * scale


_GLOBAL_NUMPY_SCALAR = np.int32(1)


def _numpy_scalar_transform(value):
    return np.asarray(value, dtype=float) * _GLOBAL_NUMPY_SCALAR.dtype.itemsize


_GLOBAL_DTYPE_ARRAY = np.array([1], dtype=np.dtype("i4"))


def _dtype_metadata_transform(value):
    metadata = _GLOBAL_DTYPE_ARRAY.dtype.metadata or {"scale": 0}
    return np.asarray(value, dtype=float) * metadata["scale"]


def _portable_fit(_value, _labels):
    return None


def test_callable_extractor_validates_numeric_2d_output():
    extractor = CallableExtractor(
        "stats",
        lambda X: np.column_stack([np.mean(X, axis=1), np.std(X, axis=1)]),
        modality="tabular",
        recipe_data={"features": ["mean", "std"]},
    )

    output = extractor.fit_transform(np.arange(12, dtype=float).reshape(4, 3))

    assert output.shape == (4, 2)
    assert extractor.recipe()["recipe_data"] == {"features": ["mean", "std"]}


def test_callable_extractor_rejects_1d_output():
    extractor = CallableExtractor("bad", lambda X: np.asarray([1, 2, 3]))

    with pytest.raises(ValueError, match="2D"):
        extractor.transform([[1], [2], [3]])


def test_callable_cache_identity_requires_portable_code_or_explicit_opt_in():
    portable = CallableExtractor("portable", _portable_transform)
    local = CallableExtractor("local", lambda value: np.asarray(value))
    explicit = CallableExtractor(
        "explicit", lambda value: np.asarray(value), cache_identity="custom-v1"
    )

    assert portable.recipe()["cache_safe"] is True
    assert portable.recipe()["callable_identities"]["transform_fn"]["sha256"]
    assert local.recipe()["cache_safe"] is False
    assert explicit.recipe()["cache_safe"] is True


def test_callable_cache_identity_rejects_different_bound_instance_state():
    first = CallableExtractor("first", _StatefulTransform(1.0).transform)
    second = CallableExtractor("second", _StatefulTransform(2.0).transform)
    explicit = CallableExtractor(
        "explicit",
        _StatefulTransform(2.0).transform,
        cache_identity="stateful-transform-v2",
    )

    assert first.recipe()["callable_identities"]["transform_fn"] is None
    assert second.recipe()["callable_identities"]["transform_fn"] is None
    assert first.recipe()["cache_safe"] is False
    assert second.recipe()["cache_safe"] is False
    assert explicit.recipe()["cache_safe"] is True


def test_explicit_callable_instance_identity_has_no_address_dependent_recipe_fields():
    first = CallableExtractor(
        "instance",
        _StatefulTransform(1.0),
        cache_identity="stateful-instance-v1",
    ).recipe()
    second = CallableExtractor(
        "instance",
        _StatefulTransform(9.0),
        cache_identity="stateful-instance-v1",
    ).recipe()

    assert first == second
    assert "0x" not in first["transform_fn"]


def test_callable_cache_identity_tracks_referenced_global_configuration():
    global _GLOBAL_SCALE

    original = _GLOBAL_SCALE
    try:
        _GLOBAL_SCALE = 1.0
        first = CallableExtractor("global", _global_config_transform).recipe()
        _GLOBAL_SCALE = 7.0
        second = CallableExtractor("global", _global_config_transform).recipe()
    finally:
        _GLOBAL_SCALE = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_rejects_opaque_referenced_global_state():
    extractor = CallableExtractor("global-model", _global_model_transform)

    assert extractor.recipe()["callable_identities"]["transform_fn"] is None
    assert extractor.recipe()["cache_safe"] is False


def test_callable_cache_identity_rejects_referenced_mutable_class_state():
    original = _MutableGlobalConfig.scale
    try:
        _MutableGlobalConfig.scale = 1.0
        first = CallableExtractor("global-class", _global_class_transform).recipe()
        _MutableGlobalConfig.scale = 7.0
        second = CallableExtractor("global-class", _global_class_transform).recipe()
    finally:
        _MutableGlobalConfig.scale = original

    assert first["callable_identities"]["transform_fn"] is None
    assert second["callable_identities"]["transform_fn"] is None
    assert first["cache_safe"] is False
    assert second["cache_safe"] is False


def test_callable_cache_identity_tracks_mutable_function_attributes():
    original = _function_attribute_dependency.scale
    try:
        _function_attribute_dependency.scale = 1.0
        first = CallableExtractor("function-attribute", _function_attribute_transform).recipe()
        _function_attribute_dependency.scale = 7.0
        second = CallableExtractor("function-attribute", _function_attribute_transform).recipe()
    finally:
        _function_attribute_dependency.scale = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_rejects_self_referential_runtime_metadata():
    original = dict(_self_metadata_transform.__annotations__)
    try:
        _self_metadata_transform.__annotations__ = {}
        first = CallableExtractor("self-metadata", _self_metadata_transform).recipe()
        _self_metadata_transform.__annotations__ = {"changed": "yes"}
        second = CallableExtractor("self-metadata", _self_metadata_transform).recipe()
    finally:
        _self_metadata_transform.__annotations__ = original

    assert first["callable_identities"]["transform_fn"] is None
    assert second["callable_identities"]["transform_fn"] is None
    assert first["cache_safe"] is False
    assert second["cache_safe"] is False


@pytest.mark.parametrize("state_type", [_ToNumpyState, _ArrayProtocolState])
def test_callable_cache_identity_does_not_coerce_opaque_array_like_globals(state_type):
    global _GLOBAL_DUCK_STATE

    original = _GLOBAL_DUCK_STATE
    try:
        _GLOBAL_DUCK_STATE = state_type(3.0)
        recipe = CallableExtractor("duck-state", _duck_state_transform).recipe()
    finally:
        _GLOBAL_DUCK_STATE = original

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_cache_identity_hashes_global_path_content(tmp_path):
    global _GLOBAL_CHECKPOINT_PATH

    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"first")
    original = _GLOBAL_CHECKPOINT_PATH
    try:
        _GLOBAL_CHECKPOINT_PATH = checkpoint
        first = CallableExtractor("global-path", _global_path_transform).recipe()
        checkpoint.write_bytes(b"second")
        second = CallableExtractor("global-path", _global_path_transform).recipe()
    finally:
        _GLOBAL_CHECKPOINT_PATH = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_hashes_default_path_content(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"first")
    original = _default_path_transform.__defaults__
    try:
        _default_path_transform.__defaults__ = (checkpoint,)
        first = CallableExtractor("default-path", _default_path_transform).recipe()
        checkpoint.write_bytes(b"second")
        second = CallableExtractor("default-path", _default_path_transform).recipe()
    finally:
        _default_path_transform.__defaults__ = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_tracks_referenced_live_module_attributes():
    global _MODULE_SCALE

    original = _MODULE_SCALE
    try:
        _MODULE_SCALE = 1.0
        first = CallableExtractor("module-state", _module_attribute_transform).recipe()
        _MODULE_SCALE = 9.0
        second = CallableExtractor("module-state", _module_attribute_transform).recipe()
    finally:
        _MODULE_SCALE = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_rejects_mutable_alias_topology():
    global _GLOBAL_ALIAS_CONFIG

    original = _GLOBAL_ALIAS_CONFIG
    try:
        _GLOBAL_ALIAS_CONFIG = [[], []]
        distinct = CallableExtractor("distinct", _alias_sensitive_transform).recipe()
        shared = []
        _GLOBAL_ALIAS_CONFIG = [shared, shared]
        aliased = CallableExtractor("aliased", _alias_sensitive_transform).recipe()
    finally:
        _GLOBAL_ALIAS_CONFIG = original

    assert distinct["callable_identities"]["transform_fn"] is None
    assert distinct["cache_safe"] is False
    assert aliased["callable_identities"]["transform_fn"] is None
    assert aliased["cache_safe"] is False


def test_callable_cache_identity_tracks_owned_ndarray_layout():
    global _GLOBAL_ARRAY_STATE

    original = _GLOBAL_ARRAY_STATE
    try:
        _GLOBAL_ARRAY_STATE = np.array([[1.0, 2.0], [3.0, 4.0]], order="C")
        c_order = CallableExtractor("c-order", _array_layout_transform).recipe()
        _GLOBAL_ARRAY_STATE = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        f_order = CallableExtractor("f-order", _array_layout_transform).recipe()
    finally:
        _GLOBAL_ARRAY_STATE = original

    assert c_order["cache_safe"] is True
    assert f_order["cache_safe"] is True
    assert c_order["callable_identities"] != f_order["callable_identities"]


def test_callable_cache_identity_rejects_ndarray_subclasses(tmp_path):
    global _GLOBAL_ARRAY_STATE

    mapped_path = tmp_path / "mapped.bin"
    mapped = np.memmap(mapped_path, dtype=float, mode="w+", shape=(2, 2))
    mapped[:] = [[1.0, 2.0], [3.0, 4.0]]
    original = _GLOBAL_ARRAY_STATE
    try:
        _GLOBAL_ARRAY_STATE = mapped
        recipe = CallableExtractor("memmap", _array_layout_transform).recipe()
    finally:
        _GLOBAL_ARRAY_STATE = original
        del mapped

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_cache_identity_tracks_live_code_object_replacement():
    original = _runtime_code_transform.__code__
    try:
        first = CallableExtractor("runtime-code", _runtime_code_transform).recipe()
        _runtime_code_transform.__code__ = _replacement_runtime_code.__code__
        second = CallableExtractor("runtime-code", _runtime_code_transform).recipe()
    finally:
        _runtime_code_transform.__code__ = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


def test_callable_cache_identity_tracks_canonical_builtins_and_rejects_monkeypatch(
    monkeypatch,
):
    first = CallableExtractor("builtin", _builtin_transform).recipe()
    monkeypatch.setattr(builtins, "abs", lambda value: value * 10)
    second = CallableExtractor("builtin", _builtin_transform).recipe()

    assert first["cache_safe"] is True
    assert first["callable_identities"]["transform_fn"]["builtins"]["abs"]
    assert second["callable_identities"]["transform_fn"] is None
    assert second["cache_safe"] is False


def test_callable_cache_identity_rejects_dynamic_global_lookup():
    recipe = CallableExtractor("dynamic-global", _dynamic_global_transform).recipe()

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_cache_identity_rejects_local_imports():
    recipe = CallableExtractor("local-import", _local_import_transform).recipe()

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_cache_identity_preserves_decimal_representation():
    global _GLOBAL_DECIMAL_SCALE

    original = _GLOBAL_DECIMAL_SCALE
    try:
        _GLOBAL_DECIMAL_SCALE = Decimal("1.0")
        first = CallableExtractor("decimal", _decimal_representation_transform).recipe()
        _GLOBAL_DECIMAL_SCALE = Decimal("1.00")
        second = CallableExtractor("decimal", _decimal_representation_transform).recipe()
    finally:
        _GLOBAL_DECIMAL_SCALE = original

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["callable_identities"] != second["callable_identities"]


@pytest.mark.parametrize(
    "aware",
    [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")),
    ],
)
def test_callable_cache_identity_rejects_aware_datetime_state(aware):
    global _GLOBAL_AWARE_DATETIME

    original = _GLOBAL_AWARE_DATETIME
    try:
        _GLOBAL_AWARE_DATETIME = aware
        recipe = CallableExtractor("aware-datetime", _aware_datetime_transform).recipe()
    finally:
        _GLOBAL_AWARE_DATETIME = original

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_cache_identity_preserves_numpy_scalar_dtype():
    global _GLOBAL_NUMPY_SCALAR

    original = _GLOBAL_NUMPY_SCALAR
    try:
        _GLOBAL_NUMPY_SCALAR = np.int32(1)
        int32_recipe = CallableExtractor("np-scalar", _numpy_scalar_transform).recipe()
        _GLOBAL_NUMPY_SCALAR = np.int64(1)
        int64_recipe = CallableExtractor("np-scalar", _numpy_scalar_transform).recipe()
    finally:
        _GLOBAL_NUMPY_SCALAR = original

    assert int32_recipe["cache_safe"] is True
    assert int64_recipe["cache_safe"] is True
    assert int32_recipe["callable_identities"] != int64_recipe["callable_identities"]


def test_callable_cache_identity_rejects_numpy_dtype_metadata():
    global _GLOBAL_DTYPE_ARRAY

    original = _GLOBAL_DTYPE_ARRAY
    try:
        _GLOBAL_DTYPE_ARRAY = np.array(
            [1],
            dtype=np.dtype("i4", metadata={"scale": 1}),
        )
        recipe = CallableExtractor("dtype-metadata", _dtype_metadata_transform).recipe()
    finally:
        _GLOBAL_DTYPE_ARRAY = original

    assert recipe["callable_identities"]["transform_fn"] is None
    assert recipe["cache_safe"] is False


def test_callable_fit_function_requires_explicit_fitted_state_identity():
    unsafe = CallableExtractor("fitted", _portable_transform, fit_fn=_portable_fit)
    explicit = CallableExtractor(
        "fitted",
        _portable_transform,
        fit_fn=_portable_fit,
        cache_identity="fitted-state-v1",
    )

    assert unsafe.recipe()["cache_safe"] is False
    assert explicit.recipe()["cache_safe"] is True


def test_local_path_identity_rejects_symlinked_checkpoint_content(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    target = tmp_path / "weights-v1.bin"
    target.write_bytes(b"version-one")
    linked_weight = bundle / "weights.bin"
    alias = tmp_path / "bundle-alias"
    try:
        linked_weight.symlink_to(target)
        alias.symlink_to(bundle, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")

    assert path_content_identity(bundle) is None
    assert path_content_identity(alias) is None


def test_directory_content_identity_uses_unambiguous_typed_records(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"x\0b\0y")
    (second / "a").write_bytes(b"x")
    (second / "b").write_bytes(b"y")

    first_identity = path_content_identity(first)
    second_identity = path_content_identity(second)

    assert first_identity is not None
    assert second_identity is not None
    assert first_identity["sha256"] != second_identity["sha256"]


def test_names_cache_identities_and_output_spec_names_are_normalized():
    extractor = CallableExtractor(
        "  normalized  ", _portable_transform, cache_identity="  identity-v1  "
    )

    assert extractor.name == "normalized"
    assert extractor.cache_identity == "identity-v1"
    assert extractor.recipe()["cache_identity"] == "identity-v1"
    assert EmbeddingOutputSpec("  pooled  ").name == "pooled"
    assert StructuredOutputSpec("  tokens  ", "token").name == "tokens"
    assert SpatialOutputSpec("  patches  ", SpatialLayout(1, 1)).name == "patches"


def test_precomputed_structured_and_spatial_cache_identity_tracks_nested_callables():
    structured = PrecomputedStructuredExtractor([StructuredOutputSpec("tokens", "token")])
    spatial = PrecomputedSpatialExtractor([SpatialOutputSpec("patches", SpatialLayout(1, 1))])
    unsafe_spatial = PrecomputedSpatialExtractor(
        [
            SpatialOutputSpec(
                "patches",
                SpatialLayout(1, 1),
                annotation_transform=lambda value: value,
            )
        ]
    )

    assert structured.recipe()["cache_safe"] is True
    assert spatial.recipe()["cache_safe"] is True
    assert unsafe_spatial.recipe()["cache_safe"] is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HFAudioExtractor("audio", "model", outputs=[{"name": "x", "hidden_layer": True}]),
        lambda: HFTextExtractor("text", "model", outputs=[{"name": "x", "hidden_layer": 1.5}]),
        lambda: HFVisionExtractor("vision", "model", outputs=[{"name": 7, "hidden_layer": 1}]),
        lambda: HFVideoExtractor("video", "model", outputs=[{"name": "x", "metadata": None}]),
        lambda: HFTimeSeriesExtractor(
            "series", "model", outputs=[{"name": "x", "metadata": [("a", 1)]}]
        ),
        lambda: HFMultimodalExtractor(
            "multi",
            "model",
            input_modalities={"image": "image"},
            outputs=[
                {
                    "name": "x",
                    "source": "image",
                    "model_output": "last_hidden_state",
                    "hidden_layer": True,
                }
            ],
        ),
    ],
)
def test_huggingface_output_parsers_reject_lossy_spec_coercions(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "spec_factory",
    [
        lambda metadata: EmbeddingOutputSpec("embedding", metadata=metadata),
        lambda metadata: StructuredOutputSpec("tokens", "token", metadata=metadata),
        lambda metadata: SpatialOutputSpec("patches", SpatialLayout(1, 1), metadata=metadata),
    ],
)
@pytest.mark.parametrize("metadata", [{"value": np.nan}, {"value": object()}, {"fn": lambda: 1}])
def test_output_specs_reject_unstable_or_nonfinite_metadata(spec_factory, metadata):
    with pytest.raises(ValueError, match="deterministic, finite"):
        spec_factory(metadata)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EmbeddingOutputSpec("x", hidden_layer=True),
        lambda: StructuredOutputSpec("x", "token", hidden_layer=1.5),
        lambda: SpatialOutputSpec("x", SpatialLayout(1, 1), hidden_layer=True),
        lambda: SpatialLayout(True, 1),
        lambda: SpatialLayout(1.0, 1),
        lambda: SpatialLayout(1, 1, special_tokens=False),
    ],
)
def test_direct_output_specs_and_layouts_require_exact_integer_fields(factory):
    with pytest.raises(TypeError):
        factory()


def test_callable_structured_and_spatial_adapters_reject_extra_named_outputs():
    spatial = CallableSpatialExtractor(
        "spatial",
        lambda value: {"patches": value, "extra": value},
        [SpatialOutputSpec("patches", SpatialLayout(1, 1))],
    )
    structured = CallableStructuredExtractor(
        "structured",
        lambda value: {"tokens": value, "extra": value},
        [StructuredOutputSpec("tokens", "token")],
    )

    with pytest.raises(ValueError, match="extra=.*extra"):
        spatial.transform_spatial(np.ones((2, 1, 1, 3)))
    with pytest.raises(ValueError, match="extra=.*extra"):
        structured.transform_structured(np.ones((2, 3, 4)))
