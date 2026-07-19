"""Read-only reconciliation between the local projection, native snapshots and
curated Paisa account balances.

This module is a **pure read**. It never writes a core row, never writes a
correction, and never mutates the journal. Its only job is to assemble a
side-by-side diagnostic so an operator can see where the dashboard's view of an
account and Paisa's view diverge, and *why*.

Join rules (enforced here, not elsewhere):

* The native ↔ Paisa comparison joins **only** through explicit
  ``paisa.account_mappings`` (dashboard account id → ledger account name). There
  is **no fuzzy matching**: an account without an explicit mapping is labelled
  "no mapping" rather than guessed, so a wrong pairing can never be presented as
  a confirmed balance.
* The Paisa side is sourced from the curated ``/api/assets/balance`` rollup
  (and liabilities balance for liability accounts), joined by the mapped ledger
  name. When no reliable upstream endpoint carries a balance, the Paisa cell is
  labelled unavailable — never fabricated.
* Native balances are the latest dashboard-owned ``BalanceSnapshot`` rows.
* Projected balances are computed from the projection's opening balances plus
  its emitted postings (the same data the journal is rendered from), so the
  number matches what Paisa would compute from the include file.

Anything unavailable is clearly labelled; an upstream/native mismatch is
reported as a delta, never silently "fixed".
"""

import datetime
import logging
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, BalanceSnapshot
from financial_dashboard.schemas.extensions import (
    PaisaReconcileAccountRow,
    PaisaReconcileMappingSuggestion,
    PaisaReconcileProjectionDiag,
    PaisaReconcileResponse,
)
from financial_dashboard.services.paisa.accounting import account_kind, resolve_account
from financial_dashboard.services.paisa.config import PaisaProjectionConfig, load_config
from financial_dashboard.services.paisa.orchestrator import preview
from financial_dashboard.services.paisa.renderers.base import CARD_PAYMENT_CLEARING

logger = logging.getLogger(__name__)

#: A snapshot older than this (days) is flagged stale so an operator does not
#: mistake a months-old balance for current.
NATIVE_STALE_DAYS = 45

#: An opening balance struck more than this many days before the cutover leaves
#: an unprojected gap; the limitation is surfaced rather than papered over.
OPENING_GAP_DAYS = 45

_Q2 = Decimal("0.01")


def _norm_account(name: str) -> str:
    """Backend-agnostic account-name key: lowercase alphanumerics only.

    The card-clearing account renders as ``Liabilities:Credit Card`` under the
    ledger family but ``Liabilities:CreditCard`` under beancount (spaces are
    illegal there). Normalizing to lowercase alphanumerics lets a single
    comparison recognize the same logical account across backends.
    """
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


#: Normalized form of :data:`CARD_PAYMENT_CLEARING`, so the unresolved-clearing
#: note can recognize the generic liability a card maps to under any backend.
_CLEARING_KEY = _norm_account(CARD_PAYMENT_CLEARING)


class _ProjectedDelta(NamedTuple):
    """The projected INR delta and excluded foreign posting counts for one
    ledger account.

    The displayed ``amount`` is always the INR leg only (the reconciliation is
    an INR view). ``foreign_posting_counts`` is sorted by commodity so the
    exclusion diagnostic is stable regardless of projection entry order. No FX
    conversion is performed, so foreign legs are excluded rather than silently
    added or guessed.
    """

    amount: Decimal
    foreign_posting_counts: tuple[tuple[str, int], ...]

    @property
    def has_foreign_commodity(self) -> bool:
        return bool(self.foreign_posting_counts)


def _q(value: Decimal) -> Decimal:
    return value.quantize(_Q2)


def _to_decimal_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    try:
        return str(_q(Decimal(value)))
    except InvalidOperation, TypeError, ValueError:
        return None


def _skipped_reason_counts(skipped) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in skipped or ():
        counts[row.reason] += 1
    return dict(sorted(counts.items()))


def _projection_diag(report) -> PaisaReconcileProjectionDiag | None:
    if report is None:
        return None
    return PaisaReconcileProjectionDiag(
        emitted_count=report.emitted_count,
        unknown_count=report.unknown_count,
        unmatched_count=report.unmatched_count,
        non_inr_count=report.non_inr_count,
        missing_fx_rate_count=report.missing_fx_rate_count,
        card_side_payments=report.card_side_payments,
        skipped_reason_counts=_skipped_reason_counts(report.skipped),
        investment_lot_count=report.investment_lot_count,
        investment_excluded=list(report.investment_excluded),
        investment_disposal_unresolved_count=len(report.investment_disposal_unresolved),
        # Computed diagnostics mirrored from the projection summary so the
        # reconciliation view surfaces the same per-kind/card/FX/funding counts.
        # Every field is always present on the typed ProjectionReport NamedTuple,
        # so direct attribute access is correct (no defensive getattr needed).
        imprecise_count=report.imprecise_count,
        card_payments=report.card_payments,
        card_payments_resolved=report.card_payments_resolved,
        card_payments_unresolved=report.card_payments_unresolved,
        card_payments_ambiguous_mask=report.card_payments_ambiguous_mask,
        investment_funding_remapped=report.investment_funding_remapped,
        investment_funding_unresolved=list(report.investment_funding_unresolved),
        investment_current_valuation_count=report.investment_current_valuation_count,
        investment_market_price_count=report.investment_market_price_count,
        investment_market_price_conflicts=list(
            report.investment_market_price_conflicts
        ),
        investment_value_only_count=report.investment_value_only_count,
        investment_quantity_mismatch_count=(report.investment_quantity_mismatch_count),
        investment_missing_market_price_count=(
            report.investment_missing_market_price_count
        ),
        investment_valuation_sources=list(report.investment_valuation_sources),
        cas_portfolio_count=report.cas_portfolio_count,
        cas_portfolio_labels=list(report.cas_portfolio_labels),
        cas_investment_scope=report.cas_investment_scope,
        manual_asset_count=report.manual_asset_count,
        manual_asset_labels=list(report.manual_asset_labels),
        manual_liability_count=report.manual_liability_count,
        manual_liability_labels=list(report.manual_liability_labels),
        net_worth_scope_complete=report.net_worth_scope_complete,
        kind_counts=dict(report.kind_counts),
        projected_foreign_count=report.projected_foreign_count,
        source_currencies=list(report.source_currencies),
    )


def _balance_by_posting_account(report) -> dict[str, _ProjectedDelta]:
    """Sum every posting amount per ledger account across the projection.

    A dashboard account's ledger leaf receives its bank-side postings under the
    exact name; summing the exact name yields that account's projected delta. No
    currency conversion is performed — foreign-currency postings live in their
    own commodity and are summed per the projection's own commodity tagging, so
    an INR bank balance is never polluted by a foreign entry.

    Returns the per-account INR total (the reconciliation is an INR view) plus a
    flag noting whether the account also carried non-INR legs: those are
    excluded from the displayed number, and the caller surfaces a note so the
    omission is never mistaken for completeness.
    """
    totals: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    posting_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in report.entries:
        for posting in entry.postings:
            raw_commodity = posting.commodity
            commodity = (
                "INR"
                if raw_commodity is None or not str(raw_commodity).strip()
                else str(raw_commodity).strip()
            )
            totals[posting.account][commodity] += Decimal(posting.amount)
            posting_counts[posting.account][commodity] += 1
    # Collapse to the per-account INR total. Foreign-only activity contributes
    # zero: it must never be dimensionally added to an INR opening. Keep sorted
    # foreign posting counts solely for the explicit exclusion diagnostic.
    out: dict[str, _ProjectedDelta] = {}
    for account, by_ccy in totals.items():
        foreign_counts = tuple(
            sorted(
                (commodity, count)
                for commodity, count in posting_counts[account].items()
                if commodity != "INR"
            )
        )
        out[account] = _ProjectedDelta(
            amount=by_ccy.get("INR", Decimal("0")),
            foreign_posting_counts=foreign_counts,
        )
    return out


def _latest_snapshots(
    snapshots: list[BalanceSnapshot],
) -> dict[int, BalanceSnapshot]:
    """Newest snapshot per account_id."""
    best: dict[int, BalanceSnapshot] = {}
    for snap in snapshots:
        if snap.account_id is None:
            continue
        current = best.get(snap.account_id)
        if current is None or (snap.as_of_date, snap.id) > (
            current.as_of_date,
            current.id,
        ):
            best[snap.account_id] = snap
    return best


def _resolve_ledger_name(account: Account, config: PaisaProjectionConfig) -> str:
    """Return the exact account name the configured projection resolves.

    Account declarations exist independently of opening balances, so resolving
    through the projection's own kind/backend-aware helper keeps accounts with
    no opening visible. Operator mappings retain the same precedence and strict
    backend validation they have during an actual projection.
    """
    return resolve_account(account, config.account_mappings, config.ledger_cli).name


def _paisa_balance_for(
    ledger_name: str | None,
    asset_map: dict[str, Decimal],
    liability_map: dict[str, Decimal],
) -> tuple[Decimal | None, bool]:
    """Look up the curated Paisa balance for a ledger name.

    Exact match first; if none, sum the direct child groups (``name:...``) so a
    mapped parent rolls up its children exactly the way Paisa's rollup does. No
    fuzzy/text-similarity matching is ever performed. Returns ``(balance,
    available)``; ``(None, False)`` when no reliable upstream balance exists.
    """
    if not ledger_name:
        return (None, False)
    # Asset accounts (Assets:...) and liability accounts (Liabilities:...) read
    # from their respective curated endpoints. A name we cannot classify has no
    # reliable single endpoint, so it is unavailable rather than guessed.
    head = ledger_name.split(":", 1)[0].strip().lower()
    source = None
    if head == "assets":
        source = asset_map
    elif head == "liabilities":
        source = liability_map
    if source is None:
        return (None, False)
    if ledger_name in source:
        return (source[ledger_name], True)
    prefix = ledger_name + ":"
    children = [
        v
        for grp, v in source.items()
        if grp.startswith(prefix) and ":" not in grp[len(prefix) :]
    ]
    if children:
        return (sum(children, Decimal("0")), True)
    return (None, False)


def _suggested_mapping(account: Account, config: PaisaProjectionConfig) -> str:
    """Deterministic preview-only suggested mapping for an unmapped account.

    The suggestion is the projection's default for this account kind and
    backend, resolved with an empty mapping set. It is a *preview* only — it is
    never written without an explicit accept through the normal config-save
    path.
    """
    return resolve_account(account, {}, config.ledger_cli).name


async def build_reconciliation(
    session: AsyncSession,
    *,
    config: PaisaProjectionConfig | None = None,
    asset_report=None,
    liability_report=None,
    upstream_available: bool = False,
) -> PaisaReconcileResponse:
    """Assemble the reconciliation view. Read-only; never writes a core row.

    ``asset_report``/``liability_report`` are the already-cached, normalized
    curated reports (typed NamedTuples from integrations.paisa). The surface
    fetches them (subject to the TTL cache + mode gating) and passes them with
    the same config snapshot, so upstream identity and local mappings cannot
    drift during one request. Direct domain callers may omit ``config`` for
    backwards compatibility.
    """
    config = config if config is not None else load_config()
    mode = config.mode

    # Projection diagnostics + projected balances (project mode only; a
    # connect/disabled run has no local projection to reconcile against).
    report = None
    if config.can_project:
        try:
            previewed = await preview(session, config)
            report = previewed.report if previewed.ok else None
        except Exception:  # noqa: BLE001 — optional-extension isolation
            logger.warning("Paisa reconciliation preview failed", exc_info=True)
            report = None

    projection_diag = _projection_diag(report)
    posting_totals = _balance_by_posting_account(report) if report is not None else {}
    opening_by_id = (
        {ob.account_id: ob for ob in report.openings} if report is not None else {}
    )

    # Curated Paisa balance maps, joined ONLY by explicit mapping.
    asset_map: dict[str, Decimal] = {}
    liability_map: dict[str, Decimal] = {}
    if asset_report is not None:
        for b in asset_report.breakdowns:
            try:
                asset_map[b.group] = Decimal(b.market_amount)
            except InvalidOperation, TypeError, ValueError:
                continue
    if liability_report is not None:
        for b in liability_report.breakdowns:
            try:
                liability_map[b.group] = Decimal(b.balance_amount)
            except InvalidOperation, TypeError, ValueError:
                continue

    # Native snapshots for the selected accounts.
    selected_ids = list(config.selected_account_ids)
    native_by_account: dict[int, BalanceSnapshot] = {}
    if selected_ids:
        snaps = (
            (
                await session.execute(
                    select(BalanceSnapshot)
                    .where(BalanceSnapshot.account_id.in_(selected_ids))
                    .order_by(BalanceSnapshot.as_of_date, BalanceSnapshot.id)
                )
            )
            .scalars()
            .all()
        )
        native_by_account = _latest_snapshots(list(snaps))

    # Accounts to show: the selected ones, in stable id order.
    accounts: list[Account] = []
    if selected_ids:
        accounts = list(
            (
                await session.execute(
                    select(Account)
                    .where(Account.id.in_(selected_ids))
                    .order_by(Account.id)
                )
            )
            .scalars()
            .all()
        )

    rows: list[PaisaReconcileAccountRow] = []
    suggestions: list[PaisaReconcileMappingSuggestion] = []
    today = datetime.date.today()
    cutover = config.cutover_date

    for account in accounts:
        aid = account.id
        mapped_to = _resolve_ledger_name(account, config)
        is_liability = account_kind(account.type) == "liability"

        # Native snapshot cell.
        native = native_by_account.get(aid)
        native_balance = None
        native_as_of = None
        native_stale = None
        if native is not None:
            native_balance = _to_decimal_str(native.value)
            native_as_of = native.as_of_date.isoformat()
            native_stale = (today - native.as_of_date).days > NATIVE_STALE_DAYS

        # Projected ending balance cell.
        projected_balance = None
        projected_available = False
        note = None
        opening_available = False
        opening_source: str | None = None
        opening_as_of: str | None = None
        if report is not None and mapped_to is not None:
            opening = opening_by_id.get(aid)
            opening_available = opening is not None
            opening_source = opening.source if opening is not None else None
            opening_as_of = (
                opening.as_of.isoformat()
                if opening is not None and opening.as_of is not None
                else None
            )
            base = Decimal(opening.amount) if opening is not None else Decimal("0")
            info = posting_totals.get(mapped_to)
            delta_amount = info.amount if info is not None else Decimal("0")
            # Liabilities are projected under the ledger credit-normal
            # convention (negative = amount owed); native snapshots and Paisa
            # carry them as a positive amount owed. Negate before display so the
            # delta against Paisa/native is meaningful (and never sign-flipped).
            ending = base + delta_amount
            if is_liability:
                ending = -ending
            projected_balance = _to_decimal_str(ending)
            projected_available = True
            notes: list[str] = []
            if opening is None:
                # No reliable pre-cutover snapshot or running balance: the
                # projected balance starts from zero. Surface the gap rather
                # than inventing an opening.
                notes.append(
                    "no reliable pre-cutover snapshot or running balance; "
                    "projected balance starts from zero (opening not invented)"
                )
            elif (
                opening.as_of is not None
                and cutover is not None
                and (cutover - opening.as_of).days > OPENING_GAP_DAYS
            ):
                # An opening struck far before the cutover leaves an
                # unprojected gap between it and the cutover.
                notes.append(
                    f"opening balance is from {opening.as_of.isoformat()}, "
                    f"{(cutover - opening.as_of).days} days before the cutover; "
                    f"activity in that gap is not captured by the opening"
                )
            if info is not None and info.has_foreign_commodity:
                # The displayed projected number is the INR leg only; foreign-
                # commodity postings are excluded because no FX conversion is
                # performed. Surface the gap explicitly rather than implying the
                # number is complete.
                foreign_diagnostics = ", ".join(
                    f"{commodity}={count}"
                    for commodity, count in info.foreign_posting_counts
                )
                notes.append(
                    "projected balance shown in INR only; foreign-commodity "
                    f"postings excluded: {foreign_diagnostics}; "
                    "no FX conversion performed"
                )
            if (
                is_liability
                and report.card_payments_unresolved > 0
                and _norm_account(mapped_to) != _CLEARING_KEY
            ):
                # Unresolved card payments post to the generic clearing liability
                # (Liabilities:Credit Card), not this card's specific liability,
                # so they cannot be attributed here and are absent from this
                # card's projected balance. Surface the limitation rather than
                # imply the balance is complete — and never "correct" it.
                notes.append(
                    f"{report.card_payments_unresolved} unresolved card payment(s) "
                    f"posted to the generic {CARD_PAYMENT_CLEARING} clearing and "
                    f"cannot be attributed to this specific card, so this card's "
                    f"projected balance does not reflect them (link the bank leg's "
                    f"card_id or exact card mask to resolve)"
                )
            if notes:
                note = "; ".join(notes)

        # Paisa balance cell (explicit mapping only; no fuzzy match).
        paisa_balance_dec, paisa_available = _paisa_balance_for(
            config.account_mappings.get(str(aid)), asset_map, liability_map
        )
        paisa_balance = _to_decimal_str(paisa_balance_dec)

        # Delta: only when a projected and a Paisa balance are both present and
        # denominated the same way. Labelled, never "corrected".
        delta = None
        if (
            projected_available
            and paisa_available
            and paisa_balance_dec is not None
            and projected_balance is not None
        ):
            try:
                delta = _to_decimal_str(Decimal(projected_balance) - paisa_balance_dec)
            except InvalidOperation, TypeError, ValueError:
                delta = None

        rows.append(
            PaisaReconcileAccountRow(
                account_id=aid,
                bank=account.bank,
                label=account.label,
                type=account.type,
                mapped_to=mapped_to,
                native_balance=native_balance,
                native_as_of=native_as_of,
                native_stale=native_stale,
                projected_balance=projected_balance,
                projected_available=projected_available,
                paisa_balance=paisa_balance,
                paisa_available=paisa_available,
                delta=delta,
                note=note,
                opening_available=opening_available,
                opening_source=opening_source,
                opening_as_of=opening_as_of,
            )
        )

        # Preview-only suggestions for accounts without an explicit mapping.
        if str(aid) not in config.account_mappings:
            suggestions.append(
                PaisaReconcileMappingSuggestion(
                    account_id=aid,
                    bank=account.bank,
                    label=account.label,
                    suggested_mapping=_suggested_mapping(account, config),
                )
            )

    reason = (
        None if config.can_connect else ("disabled" if mode == "disabled" else None)
    )
    return PaisaReconcileResponse(
        ok=True,
        mode=mode,
        can_connect=config.can_connect,
        can_project=config.can_project,
        projection=projection_diag,
        accounts=rows,
        suggestions=suggestions,
        upstream_available=upstream_available,
        reason=reason,
    )


__all__ = [
    "NATIVE_STALE_DAYS",
    "OPENING_GAP_DAYS",
    "build_reconciliation",
]
