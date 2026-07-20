"""Renderer strategy registry for the Paisa projection.

A backend is an exact string id — ``ledger``, ``hledger`` or ``beancount`` —
that selects a renderer strategy. Each strategy exposes the same surface:

* ``validate_account_name(name)`` — backend-strict validation.
* ``normalize_default_account(name)`` — deterministic transform of a
  projection *default* name into the backend's legal form (identity for the
  ledger family; PascalCase for beancount).
* ``render_document(doc)`` — pure ``LedgerDocument -> str``.

Dispatch is a flat dict lookup, never plugin discovery: the set of supported
backends is exactly :data:`SUPPORTED_BACKENDS`. The compatibility facade
:mod:`financial_dashboard.services.paisa.renderer` re-exports the ledger
strategy as the v1 default so existing callers keep working unchanged.
"""

from typing import Literal, Protocol

from financial_dashboard.services.paisa.renderers.base import (
    LedgerDocument,
)

BackendId = Literal["ledger", "hledger", "beancount"]

#: The exact backend ids the projection can target.
SUPPORTED_BACKENDS: tuple[str, ...] = ("ledger", "hledger", "beancount")

#: The default backend when ``paisa.ledger_cli`` is unset/unknown.
DEFAULT_BACKEND: BackendId = "ledger"


class RendererStrategy(Protocol):
    def validate_account_name(self, name: str) -> str: ...
    def normalize_default_account(self, name: str) -> str: ...
    def render_document(self, doc: LedgerDocument) -> str: ...


# A strategy is the module's namespace; the three callables above are looked up
# as attributes. Using the module object directly keeps the registry a flat,
# static table with no instantiation.
class _Strategy:
    __slots__ = (
        "backend",
        "validate_account_name",
        "normalize_default_account",
        "render_document",
    )

    def __init__(self, backend: str, module) -> None:
        self.backend = backend
        self.validate_account_name = module.validate
        self.normalize_default_account = module.normalize_default_account
        self.render_document = module.render_document


def _build_registry() -> dict[str, _Strategy]:
    # Imported lazily so the package import stays side-effect-free and the
    # strategies can reference base without a cycle.
    from financial_dashboard.services.paisa.renderers import beancount, ledger_family

    ledger = _Strategy("ledger", ledger_family)
    return {
        "ledger": ledger,
        "hledger": _Strategy("hledger", ledger_family),
        "beancount": _Strategy("beancount", beancount),
    }


_REGISTRY: dict[str, _Strategy] = _build_registry()


def validate_backend(raw: str | None) -> BackendId:
    """Coerce a raw ``paisa.ledger_cli`` value to a supported backend id.

    An empty/unknown value falls back to :data:`DEFAULT_BACKEND` so a
    misconfigured setting fails safe to the v1 backend (ledger) rather than
    crashing projection. Callers that must refuse a mismatched upstream backend
    do so in the orchestrator, after probing the real Paisa instance.
    """
    value = (raw or "").strip().lower()
    if value == "ledger" or value == "hledger" or value == "beancount":
        return value
    return DEFAULT_BACKEND


def get_renderer(backend: str | None) -> _Strategy:
    """Return the strategy for ``backend`` (defaulting to ledger).

    Always returns a strategy — ``validate_backend`` has already coerced an
    unknown value to the default before this is reached, and the registry maps
    every supported id.
    """
    return _REGISTRY[validate_backend(backend)]


def render_document(doc: LedgerDocument, backend: str | None = DEFAULT_BACKEND) -> str:
    """Render ``doc`` with the ``backend`` strategy."""
    return get_renderer(backend).render_document(doc)


def validate_account_name(name: str, backend: str | None = DEFAULT_BACKEND) -> str:
    """Validate ``name`` against the ``backend`` strategy's grammar."""
    return get_renderer(backend).validate_account_name(name)


def normalize_default_account(name: str, backend: str | None = DEFAULT_BACKEND) -> str:
    """Deterministically transform a default account name for the backend."""
    return get_renderer(backend).normalize_default_account(name)


__all__ = [
    "DEFAULT_BACKEND",
    "BackendId",
    "RendererStrategy",
    "SUPPORTED_BACKENDS",
    "get_renderer",
    "normalize_default_account",
    "render_document",
    "validate_account_name",
    "validate_backend",
]
