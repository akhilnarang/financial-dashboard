"""Async server-side client for the Paisa REST API.

Paisa (https://github.com/ananthakumaran/paisa, v0.7.x) is a desktop
personal-finance manager that wraps ``ledger`` and exposes a REST API on
loopback (default port 7500). This module talks to that API **server-side**
from the dashboard — there is no browser, no CORS, and no assumption that the
two share a network address other than the one the operator configures.

The request/response shapes here are pinned to what
``internal/server/server.go`` actually defines:

* ``GET /api/ping`` → ``{"success": true}``.
* ``GET /api/config`` → ``{"config": {...}, "accounts": [...], ...}``; the
  config carries ``readonly``, ``ledger_cli`` and ``default_currency``.
* ``POST /api/sync`` → ``{"success": bool, "message": str}``. A readonly
  instance acknowledges with fake ``{"success": true}`` that reloads nothing,
  so readonly must be detected *before* posting.
* ``GET /api/diagnosis`` → ``{"issues": [{"level", "summary",
  "description", "details"}]}``; ``level == "danger"`` is an error,
  ``"warning"`` is informational.

Authentication is the ``X-Auth: <username>:<password>`` header Paisa's
``TokenAuthMiddleware`` checks (only when ``user_accounts`` is configured),
which is also the path its rate limiter (6/min, burst 3) gates — a real source
of HTTP 429 that the bounded retry below handles.

Security posture
----------------
The base URL is validated up front and on every request:

* Only ``http``/``https`` schemes are allowed.
* Credentials, query strings and fragments in the base URL are rejected —
  authentication travels as an explicit header, never baked into the URL.
* The host must be loopback. A non-loopback host requires ``allow_remote`` AND
  ``https`` so a misconfigured base URL cannot exfiltrate the journal in
  cleartext to an arbitrary host.
* HTTP redirects are never followed; a 3xx is a hard failure, not a silent
  hop to an attacker-controlled origin.

This is a *typed* client, not a pass-through: responses are decoded into the
small set of fields the projection actually consumes. Raw upstream JSON never
leaves this module so an upstream schema change cannot reach the projection or
the journal file.
"""

import asyncio
import email.utils
import ipaddress
import logging
import urllib.parse
from typing import Any, NamedTuple

import httpx

logger = logging.getLogger(__name__)

#: Canonical Paisa loopback default (``cmd/serve.go`` flags port 7500).
DEFAULT_BASE_URL = "http://127.0.0.1:7500"
DEFAULT_TIMEOUT_SECONDS = 15.0
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

#: Bounded 429 retry. Paisa's token-auth middleware rate-limits at 6/min with a
#: burst of 3 and returns 429 *without* a ``Retry-After`` header, so the floor
#: below is what governs that case. ``Retry-After`` is honored when present but
#: capped so a hostile or wedged server cannot park us indefinitely.
MAX_429_RETRIES = 3
MAX_RETRY_AFTER_SECONDS = 30.0
RETRY_AFTER_FLOOR_SECONDS = 0.5

#: The exact sync payload — paisa's ``/api/sync`` reloads the journal, prices
#: and portfolios independently. Projection owns the journal only;
#: prices/portfolios are left untouched so we never claim to publish data we
#: did not produce.
SYNC_PAYLOAD: dict[str, bool] = {"journal": True, "prices": False, "portfolios": False}


class PaisaError(Exception):
    """Base class for all Paisa client failures. ``code`` is a short, stable
    machine label the orchestrator branches on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ValidatedURL(NamedTuple):
    scheme: str
    host: str
    port: int | None
    path_prefix: str
    is_loopback: bool
    display: str


def _is_loopback_host(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_base_url(raw: str, *, allow_remote: bool) -> ValidatedURL:
    """Validate and canonicalize a Paisa base URL.

    Raises ``PaisaError`` on any unsupported scheme, embedded credentials,
    query/fragment, smuggled path traversal, or a disallowed host. A
    non-loopback host is allowed only when ``allow_remote`` is true AND the
    scheme is ``https`` — remote projection must never happen in cleartext.
    The returned object is the only URL shape the client ever builds a request
    from.
    """
    if not raw or not raw.strip():
        raise PaisaError("invalid_url", "Paisa base URL is empty.")
    text = raw.strip()

    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError as exc:
        raise PaisaError("invalid_url", "Paisa base URL is malformed.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise PaisaError(
            "invalid_scheme",
            f"Paisa base URL scheme {scheme!r} is not allowed "
            f"(only {sorted(ALLOWED_SCHEMES)}).",
        )

    if parsed.username or parsed.password:
        raise PaisaError(
            "credentials_in_url",
            "Paisa base URL must not carry credentials; use the auth settings.",
        )
    if parsed.query:
        raise PaisaError(
            "query_in_url", "Paisa base URL must not carry a query string."
        )
    if parsed.fragment:
        raise PaisaError("fragment_in_url", "Paisa base URL must not carry a fragment.")

    host = parsed.hostname
    if not host:
        raise PaisaError("missing_host", "Paisa base URL has no host.")
    host = host.lower()
    # ``SplitResult.port`` is a validating property: malformed/non-numeric and
    # out-of-range ports raise ValueError when accessed. Convert that stdlib
    # exception into the client's typed validation boundary so a bad setting is
    # reported as a config error rather than escaping as a route-level 500.
    try:
        port = parsed.port
    except ValueError as exc:
        raise PaisaError("invalid_port", "Paisa base URL has an invalid port.") from exc

    path = parsed.path or ""
    segments: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise PaisaError(
                "path_traversal", "Paisa base URL path must not contain '..'."
            )
        segments.append(seg)
    path_prefix = "/" + "/".join(segments) if segments else ""

    loopback = _is_loopback_host(host)
    if not loopback:
        if not allow_remote:
            raise PaisaError(
                "remote_not_allowed",
                f"Paisa base URL host {host!r} is not loopback and "
                "paisa.allow_remote is false.",
            )
        if scheme != "https":
            raise PaisaError(
                "remote_requires_https",
                f"Paisa base URL host {host!r} is not loopback; remote "
                "projection requires https.",
            )

    # Canonicalize the authority instead of preserving ``parsed.netloc``:
    # hostname casing and explicit default ports do not identify different
    # Paisa instances. IPv6 literals need brackets when reconstructed.
    display_host = f"[{host}]" if ":" in host else host
    default_port = 80 if scheme == "http" else 443
    authority = (
        display_host
        if port is None or port == default_port
        else f"{display_host}:{port}"
    )
    display = urllib.parse.urlunsplit((scheme, authority, path_prefix or "/", "", ""))
    return ValidatedURL(
        scheme=scheme,
        host=host,
        port=port,
        path_prefix=path_prefix,
        is_loopback=loopback,
        display=display,
    )


# ---------------------------------------------------------------------------
# Sanitized response types
# ---------------------------------------------------------------------------


class PaisaCapabilities(NamedTuple):
    """The small, typed subset of ``/api/config`` the projection consumes.

    ``readonly`` is load-bearing: a readonly Paisa instance acknowledges
    ``/api/sync`` with a fake success that reloads nothing, so it must be
    detected before a sync is attempted. ``ledger_cli`` gates projection —
    only the ``ledger`` backend is supported. ``default_currency`` is surfaced
    so an operator can see what Paisa will treat rows as.
    """

    ledger_cli: str | None
    readonly: bool
    default_currency: str | None


class PaisaDiagnosisIssue(NamedTuple):
    level: str  # "danger" | "warning" | <other>
    summary: str
    details: str


class PaisaDiagnosis(NamedTuple):
    """The typed subset of ``/api/diagnosis``. ``ok`` is true only when no
    ``danger``-level issues are present; warnings are surfaced but do not fail
    a sync."""

    ok: bool
    danger_count: int
    warning_count: int
    issues: tuple[PaisaDiagnosisIssue, ...]
    first_message: str | None


class PaisaSyncResult(NamedTuple):
    """Result of ``POST /api/sync``. Paisa returns ``{success, message}``;
    ``accepted`` requires BOTH a 2xx status AND ``success == true`` so a
    journal-reload failure (HTTP 200 with ``success:false``) is not mistaken
    for success. ``reason`` is the sanitized upstream message when present."""

    accepted: bool
    status_code: int
    reason: str | None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_decimal_str(value: Any) -> str:
    """Normalize a Paisa money field to a stable canonical string.

    Paisa serializes ``shopspring/decimal`` as a JSON number or a numeric string;
    both are accepted. A non-numeric or missing value falls back to ``"0"`` so a
    report row is never dropped because one field changed its wire shape. The
    value is quantized to 2 dp for display stability.
    """
    if value is None:
        return "0"
    if isinstance(value, bool):
        return "0"
    try:
        from decimal import Decimal, InvalidOperation

        d = Decimal(str(value))
        if not d.is_finite():
            return "0"
        return str(d.quantize(Decimal("0.01")))
    except InvalidOperation, ValueError, TypeError:
        return "0"


def _money_lines(raw: Any) -> tuple[PaisaMoney, ...]:
    """Normalize a Paisa ``map[string]decimal`` section into sorted
    PaisaMoney rows. Returns empty for any non-mapping shape."""
    if not isinstance(raw, dict):
        return ()
    rows: list[PaisaMoney] = []
    for key, amount in raw.items():
        text = _as_str(key)
        if not text:
            continue
        rows.append(PaisaMoney(account=text, amount=_as_decimal_str(amount)))
    return tuple(sorted(rows))


def _allocation_aggregates(raw: Any) -> tuple[PaisaMoney, ...]:
    """Normalize the allocation ``aggregates`` map into curated money rows.

    Upstream v0.7.4 maps ``account name → Aggregate`` where each ``Aggregate``
    is a struct carrying ``{date, account, market_amount, ...}``; the curated
    current balance is each entry's ``market_amount`` (not the struct itself,
    which is not a decimal). The account label comes from the struct's
    ``account`` field when present, falling back to the map key. Malformed
    entries (non-dict values) are ignored. Sorted for deterministic output.
    """
    if not isinstance(raw, dict):
        return ()
    rows: list[PaisaMoney] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        label = _as_str(value.get("account")) or _as_str(key)
        if not label:
            continue
        rows.append(
            PaisaMoney(
                account=label, amount=_as_decimal_str(value.get("market_amount"))
            )
        )
    return tuple(sorted(rows))


def _sum_budget_actual(entry: Any) -> str:
    """A budget month's ``actual`` is the Decimal sum of its account rows'
    ``actual`` values.

    Upstream v0.7.4 carries no month-level ``actual``; it must be aggregated
    from the per-account rows under ``accounts``. Malformed rows (non-dict
    accounts, non-numeric ``actual``) are ignored so a single bad row never
    drops the whole month. Quantized to 2 dp.
    """
    from decimal import Decimal, InvalidOperation

    raw_accounts = entry.get("accounts") if isinstance(entry, dict) else None
    if not isinstance(raw_accounts, list):
        return "0"
    total = Decimal("0")
    for item in raw_accounts:
        if not isinstance(item, dict):
            continue
        value = item.get("actual")
        if value is None or isinstance(value, bool):
            continue
        try:
            addend = Decimal(str(value))
        except InvalidOperation, ValueError, TypeError:
            continue
        if addend.is_finite():
            total += addend
    return str(total.quantize(Decimal("0.01")))


def _extract_capabilities(payload: Any) -> PaisaCapabilities:
    """Pull only the fields we use out of the raw config response.

    Paisa nests the config under a top-level ``config`` key. Defends against an
    upstream schema change silently widening what we treat as a capability: a
    new field the projection does not read never reaches it.
    """
    if not isinstance(payload, dict):
        return PaisaCapabilities(ledger_cli=None, readonly=False, default_currency=None)
    data: dict[str, Any] = payload
    cfg = data.get("config")
    root = cfg if isinstance(cfg, dict) else data
    return PaisaCapabilities(
        ledger_cli=_as_str(root.get("ledger_cli")),
        readonly=_as_bool(root.get("readonly")),
        default_currency=_as_str(root.get("default_currency")),
    )


def _extract_diagnosis(payload: Any) -> PaisaDiagnosis:
    """Parse ``{issues: [{level, summary, description, details}]}``.

    ``level == "danger"`` counts as an error; ``"warning"`` is informational.
    An empty/missing issues list is healthy.
    """
    if not isinstance(payload, dict):
        return PaisaDiagnosis(
            ok=True, danger_count=0, warning_count=0, issues=(), first_message=None
        )
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        return PaisaDiagnosis(
            ok=True, danger_count=0, warning_count=0, issues=(), first_message=None
        )
    danger = 0
    warning = 0
    first_danger: str | None = None
    first_warning: str | None = None
    typed: list[PaisaDiagnosisIssue] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        level = _as_str(item.get("level")) or "warning"
        summary = _as_str(item.get("summary")) or ""
        details = _as_str(item.get("details")) or ""
        typed.append(PaisaDiagnosisIssue(level=level, summary=summary, details=details))
        message = summary or details
        if level == "danger":
            danger += 1
            if first_danger is None:
                first_danger = message or "danger"
        elif level == "warning":
            warning += 1
            if first_warning is None:
                first_warning = message
    # The failing issue (a danger) is the informative one; fall back to the
    # first warning only when there are no dangers.
    first_message = first_danger or first_warning
    return PaisaDiagnosis(
        ok=danger == 0,
        danger_count=danger,
        warning_count=warning,
        issues=tuple(typed),
        first_message=first_message,
    )


# ---------------------------------------------------------------------------
# Curated report reads (v0.7.4) — typed normalization, never a raw proxy
# ---------------------------------------------------------------------------


class PaisaMoney(NamedTuple):
    account: str
    amount: str


class PaisaBudgetMonth(NamedTuple):
    month: str
    forecast: str
    actual: str
    available_this_month: str
    end_of_month_balance: str


class PaisaBudgetReport(NamedTuple):
    """Normalized ``GET /api/budget``. Upstream nests ``budgetsByMonth`` keyed
    by ``YYYY-MM``; each carries account rows plus monthly totals. Only the
    monthly totals are curated — the per-account expense postings are not proxied
    (they duplicate the journal). A month's ``actual`` is aggregated from its
    account rows' ``actual`` values (v0.7.4 carries no month-level ``actual``)."""

    months: tuple[PaisaBudgetMonth, ...]
    checking_balance: str
    available_for_budgeting: str


class PaisaAllocationTarget(NamedTuple):
    name: str
    target_percent: str
    current_percent: str


class PaisaAllocationReport(NamedTuple):
    """Normalized ``GET /api/allocation``. Upstream returns ``aggregates``
    (a ``map[name]Aggregate`` where each ``Aggregate`` carries its current
    ``market_amount``), ``aggregates_timeline`` (a large daily series, NOT
    proxied) and ``allocation_targets``. Only the targets and the current
    aggregate snapshot are curated."""

    targets: tuple[PaisaAllocationTarget, ...]
    aggregate_accounts: tuple[PaisaMoney, ...]


class PaisaRecurringSequence(NamedTuple):
    key: str
    period: str | None
    interval_days: int | None
    count: int


class PaisaRecurringReport(NamedTuple):
    """Normalized ``GET /api/recurring``. Upstream returns
    ``transaction_sequences``; the full posting lists are summarized to a count
    rather than proxied."""

    sequences: tuple[PaisaRecurringSequence, ...]


class PaisaIncomeStatementPeriod(NamedTuple):
    period: str
    starting_balance: str
    ending_balance: str
    income: tuple[PaisaMoney, ...]
    interest: tuple[PaisaMoney, ...]
    expenses: tuple[PaisaMoney, ...]
    tax: tuple[PaisaMoney, ...]
    pnl: tuple[PaisaMoney, ...]


class PaisaIncomeStatementReport(NamedTuple):
    """Normalized ``GET /api/income_statement``. Upstream returns ``yearly``
    keyed by fiscal year; the equity/liabilities maps are omitted (covered by the
    liabilities report) to avoid double-counting in a summary view."""

    periods: tuple[PaisaIncomeStatementPeriod, ...]


class PaisaLiabilityBreakdown(NamedTuple):
    group: str
    drawn_amount: str
    repaid_amount: str
    interest_amount: str
    balance_amount: str
    apr: str | None


class PaisaLiabilitiesReport(NamedTuple):
    """Normalized ``GET /api/liabilities/balance``. Upstream returns
    ``liability_breakdowns`` keyed by account group."""

    breakdowns: tuple[PaisaLiabilityBreakdown, ...]


class PaisaAssetBreakdown(NamedTuple):
    """One ``GET /api/assets/balance`` row. ``market_amount`` is the curated
    current balance for the account group; the cost/gain fields are omitted from
    the reconciliation summary (they duplicate networth)."""

    group: str
    market_amount: str


class PaisaAssetsBalanceReport(NamedTuple):
    """Normalized ``GET /api/assets/balance``. Upstream returns
    ``asset_breakdowns`` keyed by account group (with a rollup)."""

    breakdowns: tuple[PaisaAssetBreakdown, ...]


#: The report kinds the curated surface can read. Each maps to one upstream
#: endpoint and one normalizer below.
REPORT_BUDGET = "budget"
REPORT_ALLOCATION = "allocation"
REPORT_RECURRING = "recurring"
REPORT_INCOME_STATEMENT = "income_statement"
REPORT_LIABILITIES = "liabilities"
REPORT_ASSETS_BALANCE = "assets_balance"
SUPPORTED_REPORTS: frozenset[str] = frozenset(
    {
        REPORT_BUDGET,
        REPORT_ALLOCATION,
        REPORT_RECURRING,
        REPORT_INCOME_STATEMENT,
        REPORT_LIABILITIES,
        REPORT_ASSETS_BALANCE,
    }
)


def _month_key_sort(key: str) -> tuple[int, int]:
    """Sort a ``YYYY-MM`` budget key numerically; unparseable keys sort last."""
    parts = str(key).split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (9999, 99)


def normalize_budget(payload: Any) -> PaisaBudgetReport:
    if not isinstance(payload, dict):
        return PaisaBudgetReport((), "0", "0")
    raw_months = payload.get("budgetsByMonth")
    months: list[PaisaBudgetMonth] = []
    if isinstance(raw_months, dict):
        for key in sorted((str(k) for k in raw_months), key=_month_key_sort):
            entry = raw_months[key]
            if not isinstance(entry, dict):
                continue
            months.append(
                PaisaBudgetMonth(
                    month=str(key),
                    forecast=_as_decimal_str(entry.get("forecast")),
                    actual=_sum_budget_actual(entry),
                    available_this_month=_as_decimal_str(
                        entry.get("availableThisMonth")
                    ),
                    end_of_month_balance=_as_decimal_str(
                        entry.get("endOfMonthBalance")
                    ),
                )
            )
    return PaisaBudgetReport(
        months=tuple(months),
        checking_balance=_as_decimal_str(payload.get("checkingBalance")),
        available_for_budgeting=_as_decimal_str(payload.get("availableForBudgeting")),
    )


def normalize_allocation(payload: Any) -> PaisaAllocationReport:
    if not isinstance(payload, dict):
        return PaisaAllocationReport((), ())
    raw_targets = payload.get("allocation_targets")
    targets: list[PaisaAllocationTarget] = []
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            name = _as_str(item.get("name")) or ""
            if not name:
                continue
            targets.append(
                PaisaAllocationTarget(
                    name=name,
                    target_percent=_as_decimal_str(item.get("target")),
                    current_percent=_as_decimal_str(item.get("current")),
                )
            )
    raw_aggregates = payload.get("aggregates")
    aggregates = _allocation_aggregates(raw_aggregates)
    return PaisaAllocationReport(targets=tuple(targets), aggregate_accounts=aggregates)


def normalize_recurring(payload: Any) -> PaisaRecurringReport:
    if not isinstance(payload, dict):
        return PaisaRecurringReport(())
    raw = payload.get("transaction_sequences")
    sequences: list[PaisaRecurringSequence] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = _as_str(item.get("key")) or ""
            txns = item.get("transactions")
            count = len(txns) if isinstance(txns, list) else 0
            interval = item.get("interval")
            interval_days = int(interval) if isinstance(interval, int) else None
            sequences.append(
                PaisaRecurringSequence(
                    key=key,
                    period=_as_str(item.get("period")),
                    interval_days=interval_days,
                    count=count,
                )
            )
    return PaisaRecurringReport(sequences=tuple(sequences))


def normalize_income_statement(payload: Any) -> PaisaIncomeStatementReport:
    if not isinstance(payload, dict):
        return PaisaIncomeStatementReport(())
    raw_yearly = payload.get("yearly")
    periods: list[PaisaIncomeStatementPeriod] = []
    if isinstance(raw_yearly, dict):
        for period in sorted(raw_yearly.keys()):
            entry = raw_yearly[period]
            if not isinstance(entry, dict):
                continue
            periods.append(
                PaisaIncomeStatementPeriod(
                    period=str(period),
                    starting_balance=_as_decimal_str(entry.get("startingBalance")),
                    ending_balance=_as_decimal_str(entry.get("endingBalance")),
                    income=_money_lines(entry.get("income")),
                    interest=_money_lines(entry.get("interest")),
                    expenses=_money_lines(entry.get("expenses")),
                    tax=_money_lines(entry.get("tax")),
                    pnl=_money_lines(entry.get("pnl")),
                )
            )
    return PaisaIncomeStatementReport(periods=tuple(periods))


def normalize_liabilities(payload: Any) -> PaisaLiabilitiesReport:
    if not isinstance(payload, dict):
        return PaisaLiabilitiesReport(())
    raw = payload.get("liability_breakdowns")
    breakdowns: list[PaisaLiabilityBreakdown] = []
    if isinstance(raw, dict):
        for group in sorted(raw.keys()):
            entry = raw[group]
            if not isinstance(entry, dict):
                continue
            breakdowns.append(
                PaisaLiabilityBreakdown(
                    group=str(group),
                    drawn_amount=_as_decimal_str(entry.get("drawn_amount")),
                    repaid_amount=_as_decimal_str(entry.get("repaid_amount")),
                    interest_amount=_as_decimal_str(entry.get("interest_amount")),
                    balance_amount=_as_decimal_str(entry.get("balance_amount")),
                    apr=_as_decimal_str(entry.get("apr")),
                )
            )
    return PaisaLiabilitiesReport(breakdowns=tuple(breakdowns))


def normalize_assets_balance(payload: Any) -> PaisaAssetsBalanceReport:
    if not isinstance(payload, dict):
        return PaisaAssetsBalanceReport(())
    raw = payload.get("asset_breakdowns")
    breakdowns: list[PaisaAssetBreakdown] = []
    if isinstance(raw, dict):
        for group in sorted(raw.keys()):
            entry = raw[group]
            if not isinstance(entry, dict):
                continue
            breakdowns.append(
                PaisaAssetBreakdown(
                    group=str(group),
                    market_amount=_as_decimal_str(entry.get("marketAmount")),
                )
            )
    return PaisaAssetsBalanceReport(breakdowns=tuple(breakdowns))


def normalize_report(report: str, payload: Any):
    """Dispatch a raw upstream payload to the right normalizer.

    Returns the typed NamedTuple for ``report``; an unsupported ``report``
    raises ``ValueError`` (callers gate on ``SUPPORTED_REPORTS`` first).
    """
    if report == REPORT_BUDGET:
        return normalize_budget(payload)
    if report == REPORT_ALLOCATION:
        return normalize_allocation(payload)
    if report == REPORT_RECURRING:
        return normalize_recurring(payload)
    if report == REPORT_INCOME_STATEMENT:
        return normalize_income_statement(payload)
    if report == REPORT_LIABILITIES:
        return normalize_liabilities(payload)
    if report == REPORT_ASSETS_BALANCE:
        return normalize_assets_balance(payload)
    raise ValueError(f"unsupported Paisa report: {report!r}")


def _retry_after_seconds(header_value: str | None) -> float:
    """Decode a ``Retry-After`` header into seconds.

    Honors both the delta-seconds and the HTTP-date forms. Returns 0 when the
    value is absent or unparseable — callers still apply the bounded back-off
    floor so a missing header (Paisa's 429 carries none) does not produce a
    retry storm.
    """
    if not header_value:
        return 0.0
    text = header_value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except TypeError, ValueError:
        return 0.0
    if when is None:
        return 0.0
    import datetime as _dt

    delta = (when - _dt.datetime.now(_dt.UTC)).total_seconds()
    return max(0.0, delta)


class PaisaClient:
    """Async client for the subset of the Paisa REST API the dashboard uses.

    A single :class:`httpx.AsyncClient` is owned per instance and closed with
    the client. ``transport`` is injectable so tests can drive the client with
    :class:`httpx.MockTransport` without touching the network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        allow_remote: bool = False,
        auth_username: str = "",
        auth_password: str = "",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._validated = validate_base_url(base_url, allow_remote=allow_remote)
        self._auth_username = auth_username
        self._auth_password = auth_password
        self._client = httpx.AsyncClient(
            base_url=self._validated.display,
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self._validated.display

    @property
    def is_loopback(self) -> bool:
        return self._validated.is_loopback

    async def __aenter__(self) -> PaisaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal request plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._auth_username:
            # Paisa's TokenAuthMiddleware reads ``X-Auth`` as ``user:pass``
            # (SplitN on ":"), so the password is sent verbatim and the server
            # hashes it. Over loopback/HTTPS this matches the upstream auth.
            headers["X-Auth"] = f"{self._auth_username}:{self._auth_password}"
        return headers

    async def _request(
        self, method: str, path: str, *, json: Any = None
    ) -> httpx.Response:
        url = self._url_for(path)
        attempts = 0
        while True:
            try:
                response = await self._client.request(
                    method, url, json=json, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise PaisaError(
                    "unreachable", f"Paisa {method} {path} failed: {exc}"
                ) from exc

            if response.is_redirect:
                raise PaisaError(
                    "redirect_disallowed",
                    f"Paisa returned a redirect ({response.status_code}); "
                    "redirects are not followed.",
                )

            if response.status_code == 429 and attempts < MAX_429_RETRIES:
                attempts += 1
                delay = _retry_after_seconds(response.headers.get("Retry-After"))
                # Floor so a missing Retry-After (Paisa sends none) still yields.
                delay = max(
                    min(delay, MAX_RETRY_AFTER_SECONDS), RETRY_AFTER_FLOOR_SECONDS
                )
                logger.warning(
                    "Paisa 429 on %s %s; retry %d/%d after %.1fs",
                    method,
                    path,
                    attempts,
                    MAX_429_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            return response

    def _url_for(self, endpoint: str) -> str:
        # ``endpoint`` is a server-relative path like "/api/ping". Join it under
        # the validated path prefix so a reverse-proxy subpath is respected
        # while still forbidding absolute URLs smuggled in via the endpoint.
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        prefix = self._validated.path_prefix
        if endpoint.startswith(prefix + "/api/") or endpoint == prefix + "/api":
            return endpoint
        return f"{prefix}{endpoint}"

    async def _request_json(self, method: str, path: str, *, json: Any = None) -> Any:
        response = await self._request(method, path, json=json)
        if response.status_code >= 400:
            raise PaisaError(
                "http_error",
                f"Paisa {method} {path} returned {response.status_code}.",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise PaisaError(
                "bad_json", f"Paisa {method} {path} returned non-JSON body."
            ) from exc

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """``GET /api/ping`` — a liveness probe. Returns True only on a 2xx."""
        try:
            response = await self._request("GET", "/api/ping")
        except PaisaError as exc:
            if exc.code == "unreachable":
                return False
            raise
        return 200 <= response.status_code < 300

    async def fetch_config(self) -> PaisaCapabilities:
        return _extract_capabilities(await self._request_json("GET", "/api/config"))

    async def sync_journal(self) -> PaisaSyncResult:
        """``POST /api/sync`` with the exact journal-only payload.

        The body is sent verbatim as ``SYNC_PAYLOAD`` — no field is added,
        removed or reordered by the caller. ``accepted`` requires both a 2xx
        status and ``success == true``, because Paisa returns HTTP 200 with
        ``{success: false, message}`` when the journal reload itself fails. A
        readonly upstream acknowledges with fake success; detect that via
        :meth:`fetch_config` before calling.
        """
        response = await self._request("POST", "/api/sync", json=SYNC_PAYLOAD)
        status_code = response.status_code
        if not (200 <= status_code < 300):
            return PaisaSyncResult(accepted=False, status_code=status_code, reason=None)
        try:
            payload = response.json()
        except ValueError:
            return PaisaSyncResult(
                accepted=False, status_code=status_code, reason="non-JSON sync response"
            )
        if not isinstance(payload, dict):
            return PaisaSyncResult(
                accepted=False,
                status_code=status_code,
                reason="malformed sync response",
            )
        accepted = _as_bool(payload.get("success"))
        reason = _as_str(payload.get("message"))
        return PaisaSyncResult(
            accepted=accepted,
            status_code=status_code,
            reason=reason if not accepted else None,
        )

    async def diagnosis(self) -> PaisaDiagnosis:
        return _extract_diagnosis(await self._request_json("GET", "/api/diagnosis"))

    # ------------------------------------------------------------------
    # Curated report reads (v0.7.4) — read-only, normalized in this module
    # ------------------------------------------------------------------

    #: Map report kind → upstream GET path. The paths are pinned to the v0.7.4
    #: router; a shape change upstream is contained by the normalizers above.
    _REPORT_PATHS: dict[str, str] = {
        REPORT_BUDGET: "/api/budget",
        REPORT_ALLOCATION: "/api/allocation",
        REPORT_RECURRING: "/api/recurring",
        REPORT_INCOME_STATEMENT: "/api/income_statement",
        REPORT_LIABILITIES: "/api/liabilities/balance",
        REPORT_ASSETS_BALANCE: "/api/assets/balance",
    }

    async def fetch_report(self, report: str) -> Any:
        """Fetch the raw payload for a curated report kind and return the typed,
        normalized NamedTuple (see :func:`normalize_report`).

        Raises ``PaisaError`` on network/HTTP/JSON failure; the caller shapes
        that into a typed ``ok=False`` body. An unsupported ``report`` raises
        ``ValueError`` (gate on :data:`SUPPORTED_REPORTS` first).
        """
        path = self._REPORT_PATHS.get(report)
        if path is None:
            raise ValueError(f"unsupported Paisa report: {report!r}")
        payload = await self._request_json("GET", path)
        return normalize_report(report, payload)
