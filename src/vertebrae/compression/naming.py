"""Stable result names for compression variants."""

from typing import Any, Dict


def compression_variant_name(name: str, metadata: Dict[str, Any]) -> str:
    """Return a legible name that preserves the requested compression recipe."""

    method = metadata.get("method", "none")
    if method == "none":
        return name
    recipe = metadata.get("recipe") or {}
    precision = metadata.get("precision") or recipe.get("precision")
    if precision:
        return f"{name}[{method}_{precision}]"
    requested_dim = recipe.get("n_components")
    if requested_dim is not None:
        return f"{name}[{method}_{requested_dim}]"
    compressed_dim = metadata.get("compressed_dim")
    if compressed_dim is not None:
        return f"{name}[{method}_{compressed_dim}]"
    return f"{name}[{method}]"
