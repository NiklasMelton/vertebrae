"""Shared validation for explicit named adapter outputs."""

from collections.abc import Mapping, Sequence
from typing import Any, Dict

from vertebrae.extractors._identity import validate_extractor_name


def validate_named_output_mapping(
    value: Any, expected_names: Sequence[str], owner: str
) -> Dict[str, Any]:
    """Require a mapping to match declared output names exactly."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} multi-output adapters must return a mapping.")
    normalized: Dict[str, Any] = {}
    for raw_name, output in value.items():
        name = validate_extractor_name(raw_name)
        if name in normalized:
            raise ValueError(f"{owner} returned duplicate output name {name!r}.")
        normalized[name] = output
    expected = list(expected_names)
    missing = [name for name in expected if name not in normalized]
    extra = [name for name in normalized if name not in expected]
    if missing or extra:
        raise ValueError(
            f"{owner} output names must exactly match the declared specs; "
            f"missing={missing}, extra={extra}."
        )
    return normalized
