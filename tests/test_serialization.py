from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from vertebrae.utils.serialization import json_dumps_strict, make_json_safe


class Mode(str, Enum):
    TRAIN = "train"


@dataclass
class Metadata:
    path: Path
    mode: Mode
    values: np.ndarray
    tags: set[str]


def test_make_json_safe_supports_scientific_and_structured_metadata():
    value = {
        "metadata": Metadata(
            path=Path("models/example"),
            mode=Mode.TRAIN,
            values=np.asarray([1, 2], dtype=np.int64),
            tags={"beta", "alpha"},
        ),
        "tuple": (np.float32(1.5), "x"),
        "sparse": sparse.csr_matrix(np.asarray([[0.0, 2.0]])),
        "mapping": {1: "integer", "1": "string"},
    }

    normalized = make_json_safe(value)

    assert normalized["metadata"] == {
        "path": "models/example",
        "mode": "train",
        "values": [1, 2],
        "tags": ["alpha", "beta"],
    }
    assert normalized["tuple"] == [1.5, "x"]
    assert normalized["sparse"]["data"] == [2.0]
    assert len(normalized["mapping"]) == 2
    assert normalized["mapping"]["1"] == "string"


def test_json_serialization_is_deterministic_for_sets_and_mapping_keys():
    first = {"items": {"c", "a", "b"}, "mapping": {2: "b", 1: "a"}}
    second = {"mapping": {1: "a", 2: "b"}, "items": {"b", "c", "a"}}

    assert json_dumps_strict(first) == json_dumps_strict(second)


def test_make_json_safe_reports_nested_unsupported_path():
    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        make_json_safe({"metadata": {"model": object()}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_make_json_safe_rejects_non_finite_values_with_path(value):
    with pytest.raises(TypeError, match=r"non-finite float at \$\.metadata\.score"):
        make_json_safe({"metadata": {"score": value}})


def test_make_json_safe_rejects_recursive_values_with_path():
    value = []
    value.append(value)

    with pytest.raises(TypeError, match=r"recursive value at \$\[0\]"):
        make_json_safe(value)
