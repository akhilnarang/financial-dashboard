"""First-party extension framework.

Extensions are registered explicitly from this package — there is no dynamic
discovery or plugin scanning. ``register_builtin_extensions`` is called from
the app lifespan before settings are loaded so that contributed SettingDef
entries are present in services.settings when load_all_settings() runs.
"""

from financial_dashboard.extensions.base import (
    EXTENSION_CONTRACT_VERSION,
    Capability,
    ExtensionHealthMeta,
    ExtensionManifest,
    ExtensionNavItem,
    ExtensionRegistrationError,
    ExtensionRuntime,
)
from financial_dashboard.extensions.paisa import PAISA_EXTENSION
from financial_dashboard.extensions.registry import ExtensionRegistry
from financial_dashboard.services.settings import SETTINGS_REGISTRY, register_setting

BUILTIN_EXTENSIONS: tuple[ExtensionManifest, ...] = (PAISA_EXTENSION,)


def register_builtin_extensions(registry: ExtensionRegistry) -> None:
    """Register every builtin manifest into *registry* and its settings globally.

    Idempotent across restarts within a single process: manifests register into
    the (per-app) registry instance, while contributed settings are reconciled
    against the process-wide SETTINGS_REGISTRY. A setting key that is already
    present with an *equal* SettingDef is accepted; a key present with a
    *different* definition raises ExtensionRegistrationError rather than
    silently overwriting or skipping it.
    """
    for manifest in BUILTIN_EXTENSIONS:
        registry.register(manifest)
        for key, defn in manifest.settings.items():
            existing = SETTINGS_REGISTRY.get(key)
            if existing is not None:
                if existing != defn:
                    raise ExtensionRegistrationError(
                        f"Extension {manifest.id!r} contributed a conflicting "
                        f"definition for setting {key!r}"
                    )
                continue
            register_setting(key, defn)


__all__ = [
    "BUILTIN_EXTENSIONS",
    "EXTENSION_CONTRACT_VERSION",
    "Capability",
    "ExtensionHealthMeta",
    "ExtensionManifest",
    "ExtensionNavItem",
    "ExtensionRegistrationError",
    "ExtensionRegistry",
    "ExtensionRuntime",
    "PAISA_EXTENSION",
    "register_builtin_extensions",
]
