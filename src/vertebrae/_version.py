"""Runtime package version resolution."""

from importlib.metadata import PackageNotFoundError, version


def resolve_version() -> str:
    """Return the installed distribution version or a source-tree fallback."""

    try:
        return version("vertebrae")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = resolve_version()
