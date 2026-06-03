"""Helpers for optional dependency imports."""

from typing import Any


def optional_import(module_name: str, install_hint: str) -> Any:
    """Import an optional dependency with a user-friendly error message.

    Args:
        module_name: Module to import.
        install_hint: Installation instructions included in the error.

    Returns:
        Imported module.

    Raises:
        ImportError: If the optional module is unavailable.
    """

    try:
        module = __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{module_name}' is required for this feature. "
            f"Install it with: {install_hint}"
        ) from exc
    return module
