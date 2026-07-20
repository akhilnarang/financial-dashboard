"""Typed loader for the ``paisa.*`` settings.

The settings themselves live in the shared ``settings`` registry (owned by
another agent); this module only *reads* them through the existing accessors
and resolves them into a :class:`PaisaProjectionConfig` the projection and
orchestrator consume. Secrets are read through the same accessors so they
participate in the registry's encryption handling — nothing here decrypts or
stores credentials itself.

Mode semantics
--------------
``paisa.mode`` has exactly three values, and the orchestrator enforces them:

* ``disabled`` — fully inactive. No probe, no projection, no writes.
* ``connect`` — may talk to the Paisa API (ping/probe/diagnosis) but MUST NOT
  project, generate or sync. Lets an operator verify connectivity without
  enabling writes.
* ``project`` — full local projection, file generation and manual sync are
  permitted.

Backend selection
------------------
``paisa.ledger_cli`` selects the renderer backend the projection targets. It is
read here (not validated against the registry) so an unknown value still loads
a config; :func:`financial_dashboard.services.paisa.renderers.validate_backend`
coerces it to the default (ledger). On a manual sync, the orchestrator requires
the probed upstream Paisa backend to equal this configured backend and be one
of the supported ids (ledger/hledger/beancount); ``connect`` works regardless.

Non-INR policy
--------------
``paisa.non_inr_policy`` is ``skip`` (default) or ``priced``:

* ``skip`` — a non-INR transaction is never emitted; preserved exactly as v1.
* ``priced`` — a non-INR transaction whose currency has a configured
  :setting:`paisa.fx_rates` rate on/before its date is emitted as a balanced
  foreign-currency entry plus a deduplicated price directive. Without a rate it
  is skipped and reported ``missing_fx_rate``. No network price calls and no
  implicit currency conversion ever happen.

Any other configured value is safely coerced back to ``skip`` so a non-INR
amount is never emitted labelled INR.
"""

import datetime
from decimal import Decimal, DecimalException
from typing import Literal, NamedTuple

from financial_dashboard.core.dates import parse_date
from financial_dashboard.services.settings import (
    get_setting,
    get_setting_bool,
    get_setting_int,
    get_setting_json,
)

#: The backend the projection targets by default. The full supported set lives
#: in :mod:`financial_dashboard.services.paisa.renderers`; kept here as a
#: backward-compatible alias for the orchestrator's v1 import.
SUPPORTED_LEDGER_CLI = "ledger"

Mode = Literal["disabled", "connect", "project"]
NonInrPolicy = Literal["skip", "priced"]

DEFAULT_BASE_URL = "http://127.0.0.1:7500"
DEFAULT_TIMEOUT_SECONDS = 15
GENERATED_HEADER_VERSION = "3"


class FxRate(NamedTuple):
    """One historical FX quote for a currency: ``rate`` INR per unit, effective
    as of ``date``. The projection picks the latest rate on/before a
    transaction date; rates are validated positive and quantized to 4 dp."""

    date: datetime.date
    rate: Decimal


class PaisaProjectionConfig(NamedTuple):
    """Resolved, typed view of the ``paisa.*`` settings.

    ``cutover_date`` is required for projection — it is the date the opening
    balances are struck and the exclusive lower bound for projected
    transactions. ``None`` here means "not configured" and the orchestrator
    refuses to project.

    ``ledger_cli`` and ``fx_rates`` default to ``ledger`` / empty so an existing
    caller that constructs a config with only the v1 fields still works.
    """

    mode: str
    base_url: str
    external_url: str
    allow_remote: bool
    auth_username: str
    auth_password: str
    generated_path: str
    selected_account_ids: tuple[int, ...]
    cutover_date: datetime.date | None
    account_mappings: dict[str, str]
    category_mappings: dict[str, str]
    non_inr_policy: NonInrPolicy
    request_timeout_seconds: int
    ledger_cli: str = SUPPORTED_LEDGER_CLI
    fx_rates: dict[str, tuple[FxRate, ...]] = {}
    #: When true, the projection additionally emits complete investment lots
    #: (from :class:`InvestmentLot`) as conservative cost-basis opening posts.
    #: Default off; the ``paisa.project_investments`` setting registration is
    #: owned by the extension manifest, so this is read here with a false
    #: fallback for a DB that has not registered it yet.
    project_investments: bool = False

    @property
    def can_connect(self) -> bool:
        """Any mode above ``disabled`` may probe/read the Paisa API."""
        return self.mode in ("connect", "project")

    @property
    def can_project(self) -> bool:
        """Only ``project`` mode may run projection, generation or sync."""
        return self.mode == "project"

    @property
    def ready_to_project(self) -> bool:
        """A cutover date and at least one selected account are required."""
        return self.cutover_date is not None and bool(self.selected_account_ids)

    def fx_rate_for(self, currency: str, on_or_before: datetime.date) -> FxRate | None:
        """Latest configured rate for ``currency`` effective on/before the date.

        Returns ``None`` when the currency has no configured rate that covers
        the date — the projection then reports ``missing_fx_rate`` rather than
        fabricating one. ``currency`` is matched case-insensitively (normalized
        uppercase at load time). Lookup is O(log n)-ish via a linear scan over
        the per-currency list, which is tiny by construction.
        """
        rates = self.fx_rates.get(currency.upper())
        if not rates:
            return None
        chosen: FxRate | None = None
        for rate in rates:
            if rate.date <= on_or_before and (
                chosen is None or rate.date > chosen.date
            ):
                chosen = rate
        return chosen


def _coerce_mode(raw: str) -> str:
    """Normalize the mode setting to one of the three allowed values.

    An unknown or empty value is ``disabled`` so a misconfigured integration
    fails safe (inactive) rather than surprising an operator with writes.
    """
    value = (raw or "disabled").strip().lower()
    if value not in ("disabled", "connect", "project"):
        return "disabled"
    return value


def _coerce_policy(raw: str) -> NonInrPolicy:
    """``priced`` is honored; anything else (including v1 ``include``) is
    ``skip`` so a non-INR amount is never emitted labelled INR."""
    return "priced" if (raw or "").strip().lower() == "priced" else "skip"


def _split_int_list(raw: object) -> tuple[int, ...]:
    """Coerce a settings JSON value into a deduplicated, ordered tuple of ints.

    ``selected_account_ids`` is stored as JSON; it may arrive as a list of ints
    or as a list of numeric strings. Non-numeric entries are dropped rather
    than crashing the whole projection.
    """
    if not isinstance(raw, list):
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            # bool is an int subclass; reject explicitly so True/False don't
            # silently become account ids 1/0.
            continue
        if isinstance(item, int):
            value = item
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            value = int(item)
        else:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _normalize_mappings(raw: object) -> dict[str, str]:
    """Coerce a settings JSON object into a ``{str: str}`` mapping.

    Keys are stringified (account ids arrive as JSON object keys, which are
    always strings already); values must be non-empty strings. Anything else is
    dropped so a malformed override cannot inject an empty account name into the
    journal. Account-name *validity* (no newlines, comment chars, etc.) is
    enforced by the projection, which knows these are ledger account names.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        out[str(key).strip()] = text
    return out


def _parse_fx_rates(raw: object) -> dict[str, tuple[FxRate, ...]]:
    """Parse the ``paisa.fx_rates`` JSON into a normalized historical map.

    Expected shape::

        {"USD": [{"date": "2026-01-15", "rate": "83.00"}, ...], ...}

    * Currency keys are uppercased; non-string keys are dropped.
    * Each ``date`` must parse as an ISO date; unparseable entries are skipped.
    * Each ``rate`` must be a positive ``Decimal`` (a number or a numeric
      string); non-positive or unparseable entries are skipped — a bad rate
      never reaches the journal. Rates are quantized to 4 decimal places so the
      rendered price directive is stable.
    * Per currency, rates are returned sorted by date (and stable for ties), so
      :meth:`PaisaProjectionConfig.fx_rate_for` can pick the latest-on/before
      deterministically.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[FxRate, ...]] = {}
    for currency, entries in raw.items():
        if not isinstance(currency, str):
            continue
        ccy = currency.strip().upper()
        if not ccy or not isinstance(entries, list):
            continue
        parsed: list[FxRate] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            date_raw = item.get("date")
            rate_raw = item.get("rate")
            if date_raw is None or rate_raw is None:
                continue
            try:
                date = parse_date(str(date_raw))
            except ValueError, OverflowError:
                continue
            if date is None:
                continue
            try:
                rate = Decimal(str(rate_raw))
                # Check finiteness before quantizing: Infinity/NaN quantization
                # itself signals InvalidOperation. Extremely large exponents
                # can signal InvalidOperation/Overflow, while tiny positive
                # values can quantize to zero; all are invalid stored quotes.
                if not rate.is_finite() or rate <= 0:
                    continue
                rate = rate.quantize(Decimal("0.0001"))
            except DecimalException, OverflowError, ValueError, TypeError:
                continue
            if not rate.is_finite() or rate <= 0:
                continue
            parsed.append(FxRate(date=date, rate=rate))
        if not parsed:
            continue
        parsed.sort(key=lambda r: r.date)
        out[ccy] = tuple(parsed)
    return out


def load_config() -> PaisaProjectionConfig:
    """Read all ``paisa.*`` settings via the shared accessors.

    Falls back to defensible defaults when a setting is unset so a fresh DB is
    runnable without a settings round-trip. Callers that need a specific value
    (e.g. the orchestrator requires ``cutover_date`` and ``can_project``)
    re-check after loading.
    """
    raw_ids = get_setting_json("paisa.selected_account_ids", [])
    cutover_raw = get_setting("paisa.project_since", "") or ""
    cutover = parse_date(cutover_raw)

    return PaisaProjectionConfig(
        mode=_coerce_mode(get_setting("paisa.mode", "disabled") or "disabled"),
        base_url=(get_setting("paisa.base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL),
        external_url=(get_setting("paisa.external_url", "") or "").strip(),
        allow_remote=get_setting_bool("paisa.allow_remote", False),
        auth_username=(get_setting("paisa.auth_username", "") or ""),
        auth_password=(get_setting("paisa.auth_password", "") or ""),
        generated_path=(get_setting("paisa.generated_path", "") or "").strip(),
        selected_account_ids=_split_int_list(raw_ids),
        cutover_date=cutover,
        account_mappings=_normalize_mappings(
            get_setting_json("paisa.account_mappings", {})
        ),
        category_mappings=_normalize_mappings(
            get_setting_json("paisa.category_mappings", {})
        ),
        # Non-INR policy is coerced to skip|priced; any unknown value (incl. v1
        # "include") falls back to skip so a non-INR amount is never emitted
        # labelled INR.
        non_inr_policy=_coerce_policy(
            get_setting("paisa.non_inr_policy", "skip") or "skip"
        ),
        request_timeout_seconds=max(
            1, get_setting_int("paisa.request_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        ),
        ledger_cli=(
            get_setting("paisa.ledger_cli", SUPPORTED_LEDGER_CLI)
            or SUPPORTED_LEDGER_CLI
        ),
        fx_rates=_parse_fx_rates(get_setting_json("paisa.fx_rates", {})),
        project_investments=get_setting_bool("paisa.project_investments", False),
    )
