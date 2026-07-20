"""First-party extension framework: manifest/spec types and registry primitives.

There is an explicit, in-repo registry of extensions (no dynamic discovery).
Each extension contributes an immutable manifest describing its stable id,
versioning, display + navigation metadata, advertised route prefixes, health
endpoints, capability set, and any SettingDef mappings it wants registered with
services.settings.

Only first-party extensions live here; there is no plugin scanner. The set of
extensions is exactly BUILTIN_EXTENSIONS in financial_dashboard/extensions/__init__.py.

An extension MAY additionally contribute a runtime (an object implementing the
:class:`ExtensionRuntime` protocol) that participates in application lifecycle
hooks (startup / shutdown / after-fetch-cycle). Runtimes are attached to an
:class:`~financial_dashboard.services.extensions.ExtensionManager` explicitly at
bootstrap — never discovered. The manager owns their lifecycle and isolates a
failure in one extension from the others.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, Mapping, runtime_checkable

from financial_dashboard.services.settings import SettingDef

#: The extension-framework contract version this revision of the dashboard
#: implements. A manifest declares the contract version it targets via
#: ``ExtensionManifest.contract_version`` so an operator can see at a glance
#: whether a builtin was written against the current framework shape. Bumping
#: this is a framework-level change (new manifest field, new lifecycle hook,
#: etc.) — not a per-extension change.
EXTENSION_CONTRACT_VERSION = "1"


class ExtensionRegistrationError(ValueError):
    """Raised when an extension or its contributed settings can't be registered."""


class Capability:
    """Declared capability tags an extension can advertise.

    SETTING_CONTRIBUTION, HTTP_READ, and PROJECTION are exercised by the Paisa
    extension; SYNTHETIC_GENERATION is declared but not yet active.

    AUTOMATION is advertised by an extension that contributes a runtime hooking
    into the fetch cycle (e.g. automatic Paisa sync).
    """

    SETTING_CONTRIBUTION = "setting_contribution"
    HTTP_READ = "http_read"  # read account/category data from Paisa over HTTP
    PROJECTION = "projection"  # project ledger data into the Paisa model
    SYNTHETIC_GENERATION = "synthetic_generation"
    AUTOMATION = "automation"  # contributes an ExtensionRuntime (lifecycle hooks)


@dataclass(frozen=True)
class ExtensionNavItem:
    """A single navigation entry an extension contributes to the dashboard UI.

    ``path`` is a dashboard web path (e.g. ``"/extensions/paisa"``). It is
    metadata only — the route itself is mounted elsewhere — so a stale value
    can never crash the app, only render a dead link.
    """

    label: str
    path: str


@dataclass(frozen=True)
class ExtensionHealthMeta:
    """Metadata pointing at the health/metrics endpoints an extension exposes.

    ``status_path`` is the GET route that returns a typed health JSON body
    (e.g. ``"/api/extensions/paisa/status"``). ``metrics_path`` is an optional
    metrics/telemetry route. Both are metadata: the manager never calls them,
    they simply let a status surface discover where to look.
    """

    status_path: str
    metrics_path: str = ""


@dataclass(frozen=True)
class ExtensionManifest:
    """Immutable description of a first-party extension.

    id                 stable unique identifier (lowercase); registry key
    display_name       human-facing name
    description        short human-facing summary
    contract_version   extension-framework contract version this manifest
                       targets (defaults to the current EXTENSION_CONTRACT_VERSION)
    extension_version  this extension's own version (semver-ish, free-form)
    capabilities       frozenset of Capability.* tags this extension contributes
    navigation         immutable tuple of UI navigation entries
    route_prefixes     immutable tuple of route prefixes this extension mounts
                       (e.g. ``("/api/extensions/paisa", "/extensions/paisa")``)
    health             optional health/metrics endpoint metadata
    settings           immutable mapping of SettingDef contributed by this
                       extension; the registry bootstraps these into
                       SETTINGS_REGISTRY
    """

    id: str
    display_name: str
    description: str = ""
    contract_version: str = EXTENSION_CONTRACT_VERSION
    extension_version: str = "0.0.0"
    capabilities: frozenset[str] = frozenset()
    navigation: tuple[ExtensionNavItem, ...] = ()
    route_prefixes: tuple[str, ...] = ()
    health: ExtensionHealthMeta | None = None
    settings: Mapping[str, SettingDef] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "navigation", tuple(self.navigation))
        object.__setattr__(self, "route_prefixes", tuple(self.route_prefixes))
        if not isinstance(self.settings, MappingProxyType):
            object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


@runtime_checkable
class ExtensionRuntime(Protocol):
    """Optional lifecycle hooks an extension may contribute.

    All methods are best-effort and isolated by the
    :class:`~financial_dashboard.services.extensions.ExtensionManager`: a raise
    in one extension's hook is logged and swallowed, never propagated to the
    caller (the fetch loop must keep running even if an optional extension
    misbehaves).

    ``extension_id`` MUST match the id of the manifest the runtime is attached
    to; the manager rejects a mismatch at registration time.

    Lifecycle ordering (owned by the manager):

    * ``startup`` runs once during the application lifespan startup, AFTER the
      DB and settings are ready. It must NOT start long-running background work
      or external processes — extensions ride the existing FetchService loop.
    * ``after_fetch_cycle`` runs at most once per FetchService poll cycle, after
      native polling, reminders, and categorization have completed.
    * ``shutdown`` runs once during lifespan teardown. It must be idempotent and
      safe to call even if ``startup`` never ran.
    """

    extension_id: str

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def after_fetch_cycle(self) -> None: ...
