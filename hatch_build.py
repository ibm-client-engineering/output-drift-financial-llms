"""Build-time allowlist guard for the public DFAH distribution."""

from __future__ import annotations

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Remove Hatchling's forced inclusion of the internal VCS ignore file."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        if self.target_name != "sdist":
            return
        force_include = build_data.get("force_include", {})
        for source, target in tuple(force_include.items()):
            if target == ".gitignore":
                del force_include[source]
