"""Application service for curated Paisa reports and reconciliation inputs.

The HTTP-facing :mod:`financial_dashboard.services.paisa.surface` owns request
concerns such as loading settings and locating the per-app cache.  This module
owns the report workflow itself: cache identity, transient client lifecycle,
typed DTO adaptation, and the two upstream reads needed by reconciliation.

Dependencies that vary by application or test are explicit.  In particular,
the service receives its cache and fetch function instead of reaching through
``app.state`` or constructing hidden module-global collaborators.
"""

import hashlib
import logging
from typing import Any, Awaitable, Callable, NamedTuple

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.integrations.paisa import (
    PaisaAssetsBalanceReport,
    PaisaClient,
    PaisaError,
    PaisaLiabilitiesReport,
    REPORT_ALLOCATION,
    REPORT_ASSETS_BALANCE,
    REPORT_BUDGET,
    REPORT_INCOME_STATEMENT,
    REPORT_LIABILITIES,
    REPORT_RECURRING,
    validate_base_url,
)
from financial_dashboard.schemas.extensions import (
    PaisaAllocationReport,
    PaisaAllocationTarget,
    PaisaBudgetMonth,
    PaisaBudgetReport,
    PaisaIncomeStatementPeriod,
    PaisaIncomeStatementReport,
    PaisaLiabilitiesReport as PaisaLiabilitiesReportDto,
    PaisaLiabilityBreakdown,
    PaisaReconcileResponse,
    PaisaRecurringReport,
    PaisaRecurringSequence,
    PaisaReportSummary,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.reconciliation import build_reconciliation
from financial_dashboard.services.paisa.report_cache import PaisaReportCache

logger = logging.getLogger(__name__)

ReportFetcher = Callable[[PaisaProjectionConfig, str], Awaitable[Any]]
ReconciliationBuilder = Callable[..., Awaitable[PaisaReconcileResponse]]


class ReportCacheKey(NamedTuple):
    """Secret-free identity for one normalized report response source."""

    report: str
    base_url: str
    auth_username: str
    credential_fingerprint: str
    ledger_cli: str


def report_cache_key(config: PaisaProjectionConfig, report: str) -> ReportCacheKey:
    """Build a cache identity that changes with the effective connection."""
    validated = validate_base_url(config.base_url, allow_remote=config.allow_remote)
    username = config.auth_username
    # Paisa sends X-Auth only when username is non-empty.  A dormant password
    # therefore must not create a distinct effective request identity.
    password = config.auth_password if username else ""
    credential_fingerprint = hashlib.sha256(
        b"financial-dashboard:paisa-report-cache\0"
        + username.encode("utf-8")
        + b"\0"
        + password.encode("utf-8")
    ).hexdigest()
    return ReportCacheKey(
        report=report,
        base_url=validated.display,
        auth_username=username,
        credential_fingerprint=credential_fingerprint,
        ledger_cli=(config.ledger_cli or "").strip().lower(),
    )


async def fetch_report(config: PaisaProjectionConfig, report: str) -> Any:
    """Fetch and normalize one report with a short-lived Paisa client."""
    client = PaisaClient(
        base_url=config.base_url,
        allow_remote=config.allow_remote,
        auth_username=config.auth_username,
        auth_password=config.auth_password,
        timeout_seconds=float(config.request_timeout_seconds or 15),
    )
    try:
        return await client.fetch_report(report)
    finally:
        await client.aclose()


def _money(rows) -> list[dict[str, str]]:
    return [{"account": row.account, "amount": row.amount} for row in rows]


# ``assets_balance`` deliberately bypasses the report-page DTO and feeds the
# reconciliation workflow.  Keeping the supported page kinds explicit makes a
# missing adapter an invariant failure instead of a silent generic response.
DTO_REPORT_KINDS: frozenset[str] = frozenset(
    {
        REPORT_BUDGET,
        REPORT_ALLOCATION,
        REPORT_RECURRING,
        REPORT_INCOME_STATEMENT,
        REPORT_LIABILITIES,
    }
)


def report_to_dto(report: str, payload: Any) -> PaisaReportSummary:
    """Adapt one integrations-layer report into the stable API DTO."""
    if report == REPORT_BUDGET:
        return PaisaReportSummary(
            ok=True,
            report=report,
            budget=PaisaBudgetReport(
                months=[
                    PaisaBudgetMonth(
                        month=month.month,
                        forecast=month.forecast,
                        actual=month.actual,
                        available_this_month=month.available_this_month,
                        end_of_month_balance=month.end_of_month_balance,
                    )
                    for month in payload.months
                ],
                checking_balance=payload.checking_balance,
                available_for_budgeting=payload.available_for_budgeting,
            ),
        )
    if report == REPORT_ALLOCATION:
        return PaisaReportSummary(
            ok=True,
            report=report,
            allocation=PaisaAllocationReport(
                targets=[
                    PaisaAllocationTarget(
                        name=target.name,
                        target_percent=target.target_percent,
                        current_percent=target.current_percent,
                    )
                    for target in payload.targets
                ],
                aggregate_accounts=_money(payload.aggregate_accounts),
            ),
        )
    if report == REPORT_RECURRING:
        return PaisaReportSummary(
            ok=True,
            report=report,
            recurring=PaisaRecurringReport(
                sequences=[
                    PaisaRecurringSequence(
                        key=sequence.key,
                        period=sequence.period,
                        interval_days=sequence.interval_days,
                        count=sequence.count,
                    )
                    for sequence in payload.sequences
                ]
            ),
        )
    if report == REPORT_INCOME_STATEMENT:
        return PaisaReportSummary(
            ok=True,
            report=report,
            income_statement=PaisaIncomeStatementReport(
                periods=[
                    PaisaIncomeStatementPeriod(
                        period=period.period,
                        starting_balance=period.starting_balance,
                        ending_balance=period.ending_balance,
                        income=_money(period.income),
                        interest=_money(period.interest),
                        expenses=_money(period.expenses),
                        tax=_money(period.tax),
                        pnl=_money(period.pnl),
                    )
                    for period in payload.periods
                ]
            ),
        )
    if report == REPORT_LIABILITIES:
        return PaisaReportSummary(
            ok=True,
            report=report,
            liabilities=PaisaLiabilitiesReportDto(
                breakdowns=[
                    PaisaLiabilityBreakdown(
                        group=breakdown.group,
                        drawn_amount=breakdown.drawn_amount,
                        repaid_amount=breakdown.repaid_amount,
                        interest_amount=breakdown.interest_amount,
                        balance_amount=breakdown.balance_amount,
                        apr=breakdown.apr,
                    )
                    for breakdown in payload.breakdowns
                ]
            ),
        )
    raise AssertionError(
        f"report_to_dto got unsupported report {report!r}; expected one of "
        f"{sorted(DTO_REPORT_KINDS)}"
    )


class PaisaReportService:
    """Curated-report workflow with cache and upstream access injected."""

    def __init__(
        self,
        cache: PaisaReportCache,
        *,
        fetch_report_fn: ReportFetcher = fetch_report,
        reconciliation_builder: ReconciliationBuilder = build_reconciliation,
    ) -> None:
        self._cache = cache
        self._fetch_report = fetch_report_fn
        self._build_reconciliation = reconciliation_builder

    async def report_summary(
        self,
        config: PaisaProjectionConfig,
        report: str,
        *,
        ttl_seconds: int,
    ) -> PaisaReportSummary:
        """Return one typed report response with optional-extension isolation."""
        if report not in DTO_REPORT_KINDS:
            return PaisaReportSummary(
                ok=False, report=report, reason="unsupported_report"
            )
        if not config.can_connect:
            return PaisaReportSummary(ok=False, report=report, reason="disabled")

        try:
            key = report_cache_key(config, report)
            result = await self._cache.read(
                key,
                ttl_seconds,
                lambda: self._fetch_report(config, report),
            )
        except PaisaError as exc:
            return PaisaReportSummary(ok=False, report=report, reason=exc.code)
        except Exception as exc:  # noqa: BLE001 — optional-extension isolation
            return PaisaReportSummary(ok=False, report=report, reason=str(exc))

        dto = report_to_dto(report, result.value)
        dto.cached = result.hit
        return dto

    async def reconciliation_view(
        self,
        session: AsyncSession,
        config: PaisaProjectionConfig,
        *,
        ttl_seconds: int,
    ) -> PaisaReconcileResponse:
        """Build the read-only reconciliation view from curated balances."""
        asset_report: PaisaAssetsBalanceReport | None = None
        liability_report: PaisaLiabilitiesReport | None = None
        upstream_available = False
        if config.can_connect:
            try:
                asset_report = (
                    await self._cache.read(
                        report_cache_key(config, REPORT_ASSETS_BALANCE),
                        ttl_seconds,
                        lambda: self._fetch_report(config, REPORT_ASSETS_BALANCE),
                    )
                ).value
                liability_report = (
                    await self._cache.read(
                        report_cache_key(config, REPORT_LIABILITIES),
                        ttl_seconds,
                        lambda: self._fetch_report(config, REPORT_LIABILITIES),
                    )
                ).value
                upstream_available = True
            except PaisaError:
                upstream_available = False
            except Exception:  # noqa: BLE001 — optional-extension isolation
                logger.warning(
                    "Paisa reconciliation upstream read failed", exc_info=True
                )
                upstream_available = False

        return await self._build_reconciliation(
            session,
            config=config,
            asset_report=asset_report,
            liability_report=liability_report,
            upstream_available=upstream_available,
        )


__all__ = [
    "DTO_REPORT_KINDS",
    "PaisaReportService",
    "ReportCacheKey",
    "fetch_report",
    "report_cache_key",
    "report_to_dto",
]
