"""Package metadata that is safe to import from internal modules."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    """Return the installed distribution version or a source-checkout fallback."""

    try:
        return version("dfah-bench")
    except PackageNotFoundError:
        return "0.1.1.dev0"
