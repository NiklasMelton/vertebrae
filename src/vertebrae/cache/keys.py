"""Canonical artifact-key validation and named-output encoding."""

import hashlib
import re
import unicodedata
from typing import Iterable

_OUTPUT_KEY_VERSION = "v1"
_OUTPUT_SLUG_MAX_LENGTH = 40
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def validate_artifact_key(key: str) -> str:
    """Validate and return a canonical relative artifact key.

    Artifact keys use forward-slash-separated relative components. Validation is
    intentionally lossless: valid keys are returned unchanged and invalid keys
    are rejected instead of being normalized into a potentially colliding key.
    """

    if not isinstance(key, str):
        raise ValueError("Artifact key must be a string.")
    if not key:
        raise ValueError("Artifact key must be non-empty.")
    if key.startswith("/"):
        raise ValueError("Artifact key must be relative and must not start with '/'.")
    if "\\" in key:
        raise ValueError("Artifact key must use '/' separators and must not contain '\\'.")
    if any(unicodedata.category(character) == "Cc" for character in key):
        raise ValueError("Artifact key must not contain control characters.")

    components = key.split("/")
    if any(component == "" for component in components):
        raise ValueError("Artifact key must not contain empty path components.")
    if any(component in {".", ".."} for component in components):
        raise ValueError("Artifact key must not contain '.' or '..' path components.")
    return key


def named_output_key_segment(output_name: str) -> str:
    """Encode an exact output name as a readable collision-resistant segment."""

    if not isinstance(output_name, str):
        raise ValueError("Output name must be a string.")
    if not output_name:
        raise ValueError("Output name must be non-empty.")

    display_name = unicodedata.normalize("NFKD", output_name).encode(
        "ascii", errors="ignore"
    ).decode("ascii")
    slug = _NON_ALPHANUMERIC.sub("-", display_name.lower()).strip("-")
    slug = slug[:_OUTPUT_SLUG_MAX_LENGTH].rstrip("-") or "output"
    digest = hashlib.sha256(output_name.encode("utf-8")).hexdigest()
    return f"output-{_OUTPUT_KEY_VERSION}-{slug}--{digest}"


def named_output_artifact_key(base_key: str, output_name: str) -> str:
    """Build a canonical artifact key for one named output."""

    base_key = validate_artifact_key(base_key)
    key = f"{base_key}/outputs/{named_output_key_segment(output_name)}"
    return validate_artifact_key(key)


def named_output_artifact_keys(
    base_key: str,
    output_names: Iterable[str],
) -> dict[str, str]:
    """Build and validate all named-output keys before any caller writes them."""

    names = list(output_names)
    for name in names:
        if not isinstance(name, str):
            raise ValueError("Output name must be a string.")
        if not name:
            raise ValueError("Output name must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError("Output names must be unique before artifact keys are generated.")

    keys = [named_output_artifact_key(base_key, name) for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("Named outputs generated colliding artifact keys.")
    return dict(zip(names, keys))
