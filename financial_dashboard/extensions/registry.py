"""Explicit in-repo extension registry.

No dynamic/plugin discovery: extensions are registered imperatively at
startup. Registration order is insertion order (deterministic) and duplicate
ids are rejected.
"""

from collections.abc import Iterator

from financial_dashboard.extensions.base import (
    ExtensionManifest,
    ExtensionRegistrationError,
)


class ExtensionRegistry:
    """Ordered map of extension id -> manifest with collision-safe registration."""

    __slots__ = ("_extensions",)

    def __init__(self) -> None:
        self._extensions: dict[str, ExtensionManifest] = {}

    def register(self, manifest: ExtensionManifest) -> None:
        """Register a manifest. Raises ExtensionRegistrationError on duplicate id."""
        if not isinstance(manifest, ExtensionManifest):
            raise ExtensionRegistrationError(
                f"Expected ExtensionManifest, got {type(manifest).__name__}"
            )
        if manifest.id in self._extensions:
            raise ExtensionRegistrationError(
                f"Extension already registered: {manifest.id}"
            )
        self._extensions[manifest.id] = manifest

    def get(self, ext_id: str) -> ExtensionManifest | None:
        return self._extensions.get(ext_id)

    def all(self) -> tuple[ExtensionManifest, ...]:
        """Return manifests in deterministic registration order."""
        return tuple(self._extensions.values())

    def __contains__(self, ext_id: object) -> bool:
        return ext_id in self._extensions

    def __iter__(self) -> Iterator[ExtensionManifest]:
        return iter(self._extensions.values())

    def __len__(self) -> int:
        return len(self._extensions)
