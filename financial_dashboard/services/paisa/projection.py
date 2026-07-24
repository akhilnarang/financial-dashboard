"""Read-only projection of dashboard transactions onto a backend journal.

The projection is a **pure read** over the existing ORM: it SELECTs accounts,
balance snapshots and transactions, assembles a :class:`LedgerDocument`, and
hands it to the configured renderer strategy. It never INSERTs, UPDATEs or
DELETEs a core row — a failed or partial projection cannot corrupt the
dashboard's data, and the publisher is the only thing that writes, and it
writes a single include file.

Scope rules (all enforced here, none delegated):

* Only transactions linked to ``selected_account_ids`` are considered.
* Only transactions whose ``transaction_date`` is strictly after the cutover
  are emitted; the cutover itself is captured by the opening-balances entry.
* Multi-currency policy (``paisa.non_inr_policy``):
    - ``skip`` (default) — a non-INR transaction is never emitted (v1).
    - ``priced`` — a non-INR transaction whose currency has a configured
      ``paisa.fx_rates`` rate on/before its date is emitted as a balanced
      entry in that currency plus a deduplicated price directive. Without a
      rate it is skipped and reported ``missing_fx_rate``. NULL currency is an
      INR row that predates the column's default, so it is always kept.
* Self-transfers are paired by shared reference and emitted once; a lone leg
  is reported ``unmatched`` and skipped. Under ``priced``, a pair must share
  the same currency (a cross-currency pair is reported, not collapsed).
* Card payments (``credit_card_payment``) are emitted as a bank→liability
  transfer, never as an expense, so the same rupee is not counted twice.

Opening balances are always INR (requirement: never fabricate a commodity).

No network price calls and no implicit currency conversion are performed: a
foreign amount is emitted in its own commodity or skipped, never relabelled.
"""

import datetime
from decimal import Decimal
from typing import NamedTuple

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Card,
    CasUpload,
    ManualItem,
    Transaction,
)
from financial_dashboard.db.enums import SnapshotCategory
from financial_dashboard.services.investments import (
    CurrentValuation,
    get_current_valuations,
)
from financial_dashboard.services.paisa.accounting import (
    CREDIT_CARD_PAYMENT_SLUG,
    INVESTMENT_CATEGORY_SLUGS,
    KIND_CARD_PAYMENT,
    KIND_CONTRA_EXPENSE,
    KIND_EXPENSE,
    KIND_INCOME,
    KIND_INVESTMENT,
    KIND_LOT,
    KIND_OPENING,
    KIND_REPAYMENT,
    KIND_SELF_TRANSFER,
    KIND_UNKNOWN,
    KIND_VALUATION,
    ProjectionError,
    REPAYMENT_CLEARING_ACCOUNT,
    SELF_TRANSFER_SLUG,
    card_clearing_account,
    category_kind,
    contra_account,
    normalize_policy_account,
    resolve_account,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.settings import get_setting
from financial_dashboard.services.paisa.renderers import (
    render_document as render_document_for_backend,
)
from financial_dashboard.services.paisa.portfolio_identity import (
    PORTFOLIO_TOKEN_SECRET_KEY,
    normalize_portfolio_key,
    portfolio_token,
)
from financial_dashboard.services.paisa.renderers.base import (
    EQUITY_REVALUATION,
    INR,
    LedgerAccount,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    PriceDirective,
    ProjectedEntry,
    investment_valuation_account,
    sanitize_meta_value,
)

# ---------------------------------------------------------------------------
# Dashboard category → accounting taxonomy
# ---------------------------------------------------------------------------
#
# The projection roots the contra account by *category semantics*, not by
# direction alone. This is what makes reversals net correctly and keeps asset
# movements out of the P&L:
#
# * ``income`` slugs (salary/interest/other_income) → Income root (always)
# * expense slugs → Expenses root (always) — a credit on an expense slug is a
#   reversal that nets against the expense, never relabelled as Income.
# * ``refund``/``cashback_rewards`` → contra-expense: Expenses root (negative
#   Expenses on credit = money back, netting against the original spend).
# * ``investment``/``investment_redemption`` → Assets:Investments:Unallocated
#   (asset movement, not expense/income).
# * ``repayment`` → Equity:Transfers In (non-income clearing root — somebody
#   paying you back is not earned income).
# * ``self_transfer`` / ``credit_card_payment`` stay special-cased above this
#   table (a self-transfer pair is one balanced transfer; a card payment is a
#   bank→liability transfer).
#
# ``emi_loan`` and ``cash_withdrawal`` are expense slugs but *imprecise*: the
# source row does not carry the principal/interest split or the cash/loan
# account the movement settled against. Rather than fabricate a loan liability
# or a cash asset, they post to a conservative Expenses clearing root and the
# projection surfaces an ``imprecise`` diagnostic so an operator knows the
# movement is not truthfully allocated.

#: Expense slugs whose contra is imprecise (no principal/cash account in the
#: source row). They post to a conservative Expenses clearing and the report
#: carries an ``imprecise`` diagnostic rather than fabricating an account.
IMPRECISE_CATEGORY_SLUGS = frozenset({"emi_loan", "cash_withdrawal"})

DASHBOARD_KINDS = frozenset(
    {
        KIND_EXPENSE,
        KIND_INCOME,
        KIND_CONTRA_EXPENSE,
        KIND_INVESTMENT,
        KIND_REPAYMENT,
        KIND_SELF_TRANSFER,
        KIND_CARD_PAYMENT,
        KIND_OPENING,
        KIND_LOT,
        KIND_UNKNOWN,
    }
)


class SkippedRow(NamedTuple):
    """A transaction the projection deliberately did not emit, with why.

    ``reason`` is a short stable label (non_inr, missing_fx_rate,
    unmatched_self_transfer, card_side_payment, orphan, ...) so a caller can
    bucket the report."""

    txn_id: int | None
    reason: str
    detail: str


class ProjectionReport(NamedTuple):
    """The full result of a projection: the rendered journal plus a bookkeeping
    of every decision so an operator can audit what was and was not emitted.

    The multi-currency fields default to zero/empty so a caller that constructs
    a report with only the v1 fields (e.g. a serialized test fixture) still
    works."""

    journal: str
    document: LedgerDocument
    entries: tuple[ProjectedEntry, ...]
    openings: tuple[OpeningBalance, ...]
    emitted_count: int
    self_transfer_pairs: int
    card_payments: int
    card_side_payments: int
    non_inr_count: int
    unmatched_count: int
    unknown_count: int
    skipped: tuple[SkippedRow, ...]
    cutover_date: datetime.date | None
    account_ids: tuple[int, ...]
    projected_foreign_count: int = 0
    missing_fx_rate_count: int = 0
    source_currencies: tuple[str, ...] = ()
    #: LEGACY compatibility fields, always 0/empty. Paisa projects CAS as an
    #: authoritative aggregate valuation and never consumes ``InvestmentLot``
    #: rows, so no lot is emitted and no lot is suppressed. They are retained
    #: only so clients built against the earlier cost-basis projection keep
    #: deserializing; the core investment service still owns lot normalization
    #: and its own diagnostics.
    investment_lot_count: int = 0
    investment_disposal_unresolved: tuple[str, ...] = ()
    #: Projection policy diagnostics only (``valuation_only_no_cost_basis``,
    #: ``portfolio_value_unavailable``). Lot-normalization reasons belong to the
    #: core investment service, not here.
    investment_excluded: tuple[str, ...] = ()
    #: Diagnostics for the dashboard taxonomy semantics.
    #: ``imprecise_count`` is the number of emitted entries whose category is
    #: inherently imprecise (emi_loan/cash_withdrawal) — posted to a
    #: conservative Expenses clearing rather than fabricating a principal/cash
    #: account.
    imprecise_count: int = 0
    #: ``card_payments_resolved`` counts card-payment bank legs that were
    #: resolved to a specific selected liability (via explicit card_id or exact
    #: mask); ``card_payments_unresolved`` is the generic-clearing count.
    card_payments_resolved: int = 0
    card_payments_unresolved: int = 0
    #: Exact masks shared by two or more selected cards are deliberately
    #: ambiguous. Their bank payment still emits against generic clearing, but
    #: this separate count prevents that safety decision looking like a simple
    #: missing mapping.
    card_payments_ambiguous_mask: int = 0
    #: LEGACY, always 0/empty. Kept only so clients built against the earlier
    #: cost-basis projection keep deserializing. Paisa no longer matches bank
    #: legs to lots at all — see ``investment_unresolved_purchases`` /
    #: ``investment_unresolved_redemptions`` for the live diagnostics.
    investment_funding_remapped: int = 0
    investment_funding_unresolved: tuple[str, ...] = ()
    #: ``kind_counts`` is the cardinality of each ``dashboard_kind`` among
    #: emitted entries (excluding openings/lots which are structurally
    #: separate). Used by closed-population tests and operator diagnostics.
    kind_counts: dict[str, int] = {}
    #: LIVE holding-fact diagnostics, read from the latest CAS statement per
    #: portfolio. They describe the source positions behind the projected
    #: aggregate — identity-preserving, so the same ISIN in two folios or demat
    #: accounts stays two facts. ``investment_value_only_count`` is the number
    #: of active positions carrying no acquisition cost, which is why CAS is
    #: projected as a valuation rather than as lots.
    investment_current_valuation_count: int = 0
    investment_value_only_count: int = 0
    investment_valuation_sources: tuple[str, ...] = ()
    #: LEGACY compatibility fields, always 0/empty. Paisa emits no CAS
    #: market-price directives and performs no lot/quantity reconciliation, so
    #: nothing can be conflicted, mismatched or missing a price. Retained only
    #: so clients built against the earlier cost-basis projection keep
    #: deserializing.
    investment_market_price_count: int = 0
    investment_market_price_conflicts: tuple[str, ...] = ()
    investment_quantity_mismatch_count: int = 0
    investment_missing_market_price_count: int = 0
    #: Closed-population net-worth scope diagnostics. Account selection cannot
    #: select CAS portfolios or manual items, so every preview/generate/sync
    #: explicitly says whether CAS is included/excluded/partial and names all
    #: active non-account sources. Manual items remain outside projection: the
    #: model has no operator-selected ledger mapping for them, so silently
    #: inventing accounts or including every private source would be untruthful.
    cas_portfolio_count: int = 0
    cas_portfolio_labels: tuple[str, ...] = ()
    cas_investment_scope: str = "none"
    manual_asset_count: int = 0
    manual_asset_labels: tuple[str, ...] = ()
    manual_liability_count: int = 0
    manual_liability_labels: tuple[str, ...] = ()
    #: How the represented CAS value is accounted. Always ``valuation_only``
    #: when any CAS value is projected: this projection reads authoritative
    #: portfolio aggregates and never consumes InvestmentLot rows, so cost
    #: basis, capital gains and XIRR are unavailable for CAS by construction.
    cas_investment_coverage: str = "none"
    #: Always empty — retained for backward compatibility with clients built
    #: against the earlier mixed cost-basis/valuation projection.
    investment_cost_basis_portfolios: tuple[str, ...] = ()
    #: Portfolios projected from their authoritative BalanceSnapshot history.
    investment_valuation_portfolios: tuple[str, ...] = ()
    investment_valuation_entry_count: int = 0
    investment_valuation_total: Decimal = Decimal("0.00")
    #: CAS portfolios with no authoritative INR snapshot to project.
    investment_valuation_unrepresented: tuple[str, ...] = ()
    #: Bank investment legs left unresolved by policy. Purchases keep their
    #: ``Assets:Investments:Unallocated`` asset (which a CAS aggregate may also
    #: contain — a possible overlap); redemptions post to the non-income
    #: clearing so an asset account is never driven negative. Neither is netted
    #: against CAS: the source data cannot say which portfolio a bank row
    #: funded. Any non-zero count makes ``net_worth_scope_complete`` false.
    investment_unresolved_purchases: int = 0
    investment_unresolved_redemptions: int = 0
    net_worth_scope_complete: bool = True
    #: Whether every native net-worth *source* is represented, independent of
    #: whether the total is exact. ``net_worth_scope_complete`` additionally
    #: requires no unresolved investment legs.
    net_worth_sources_complete: bool = True


class FxDecision(NamedTuple):
    """Per-transaction currency resolution.

    * ``currency``/``commodity``: the normalized uppercase currency (INR for a
      native row, the explicit foreign currency otherwise).
    * ``rate``: the configured INR/unit rate effective on/before the txn date,
      or ``None`` for INR / unavailable.
    * ``skip_reason``: ``"non_inr"`` (skip policy), ``"missing_fx_rate"`` (priced
      policy with no configured rate), ``"invalid_currency"`` (a non-empty
      currency that cannot be normalized to a legal backend symbol), or ``None``
      (emit).
    """

    currency: str
    commodity: str
    rate: Decimal | None
    skip_reason: str | None


class _ResolvedCard(NamedTuple):
    """A selected Card row and the liability account it owns."""

    card_id: int
    account: LedgerAccount


class _CardResolutionMaps(NamedTuple):
    """Exact card lookups, retaining only globally unique selected masks."""

    by_card_id: dict[int, _ResolvedCard]
    by_unique_mask: dict[str, _ResolvedCard]
    ambiguous_masks: frozenset[str]


class _CardPaymentResolution(NamedTuple):
    """Resolved target plus a stable metadata/diagnostic status."""

    card: _ResolvedCard | None
    status: str


class _NetWorthScopeSources(NamedTuple):
    """Active non-Account sources that an account picker cannot select."""

    cas_portfolio_labels: tuple[str, ...]
    cas_portfolio_keys: frozenset[str]
    manual_asset_labels: tuple[str, ...]
    manual_liability_labels: tuple[str, ...]


# ---------------------------------------------------------------------------
# FX classification
# ---------------------------------------------------------------------------


def _normalize_fx_currency(raw: str | None) -> str | None:
    """Normalize a transaction currency into a deterministic, backend-safe
    uppercase symbol, or return ``None`` when it cannot be made legal.

    NULL/whitespace-only is handled by the caller (it means INR — the column's
    pre-default state). For a non-empty value, surrounding/internal whitespace
    and control characters are dropped and the remainder is uppercased ASCII
    alphanumerics — so a stray newline, tab, space, ``;``, ``{`` or non-ASCII
    byte can never corrupt a posting amount or a ``P``/``price`` directive in
    any backend. The symbol must start with a letter (every backend rejects a
    digit-led commodity), so a value that normalizes to empty or digit-led
    returns ``None`` and the caller skips it with a clear ``invalid_currency``
    diagnostic instead of emitting an invalid directive.

    A valid ISO 4217 alpha code (``usd`` / ``USD`` / ``" USD "``) survives
    unchanged as ``USD``; a control-laced variant (``"US\\nD"``) normalizes to
    the same clean symbol.
    """
    cleaned = "".join(ch for ch in str(raw) if ch.isascii() and ch.isalnum()).upper()
    if not cleaned or not cleaned[0].isalpha():
        return None
    return cleaned


def _decide_fx(txn: Transaction, config: PaisaProjectionConfig) -> FxDecision:
    """Resolve a transaction's currency, rate and skip/emit decision.

    NULL/blank currency is INR (the column's pre-default state), never an
    unknown currency. Any non-``priced`` policy (including a stray v1
    ``include``) behaves as ``skip`` so a non-INR amount is never emitted
    labelled INR.

    A non-empty currency is normalized once here (see
    :func:`_normalize_fx_currency`) so every backend — including beancount,
    which emits the commodity BARE in postings and ``price`` directives —
    receives a legal deterministic symbol. A value that cannot be normalized
    is skipped as ``invalid_currency`` so a malformed/control-laced currency
    never reaches a posting or price directive.
    """
    raw = txn.currency
    if raw is None or not str(raw).strip():
        return FxDecision(INR, INR, None, None)
    currency = _normalize_fx_currency(raw)
    if currency is None:
        return FxDecision(str(raw), str(raw), None, "invalid_currency")
    if currency == INR:
        return FxDecision(INR, INR, None, None)
    if config.non_inr_policy != "priced":
        return FxDecision(currency, currency, None, "non_inr")
    date = txn.transaction_date
    fx = config.fx_rate_for(currency, date) if date is not None else None
    if fx is None:
        return FxDecision(currency, currency, None, "missing_fx_rate")
    return FxDecision(currency, currency, fx.rate, None)


def _account_sign(direction: str) -> int:
    """Signed multiplier for the *account* posting.

    A credit (money in) increases an asset and decreases a liability; a debit
    (money out) does the opposite. Under ledger's sign convention (liabilities
    are credit-normal i.e. negative balances), both asset and liability
    accounts are *credited* (negative posting) when money leaves and *debited*
    (positive posting) when money arrives — so the sign depends on direction
    alone, not on kind.
    """
    return 1 if direction == "credit" else -1


# ---------------------------------------------------------------------------
# Opening balances
# ---------------------------------------------------------------------------


async def _opening_for_account(
    session: AsyncSession,
    account: LedgerAccount,
    cutover: datetime.date,
) -> OpeningBalance | None:
    """Derive an opening balance for one account at/before the cutover.

    Opening balances are always INR. Snapshot currency is explicit and only an
    ``INR`` row is eligible; a foreign (or malformed/NULL) snapshot is never
    relabelled. Transaction ``currency IS NULL`` is the documented pre-column
    legacy state and means INR, so only that narrow fallback treats NULL as INR.

    Preference order:
      1. The latest :class:`BalanceSnapshot` at or before the cutover — the
         authoritative point-in-time balance the net-worth pipeline already
         reconciles against.
      2. The latest pre-cutover transaction's running ``balance`` for that
         account — a defensible fallback when no snapshot exists but the bank
         exposed a running balance.

    Liability (card) snapshots are stored as the outstanding owed amount, which
    is already the right sign for a credit-normal ledger liability, so they are
    negated to land on the ledger convention. Asset snapshots are positive.
    """
    snapshot = await _latest_snapshot(session, account.account_id, cutover)
    if snapshot is not None:
        value = Decimal(snapshot.value)
        if account.kind == "liability":
            # Stored as a positive "amount owed"; ledger liability is negative.
            value = -value
        return OpeningBalance(
            account_id=account.account_id,
            account_name=account.name,
            amount=value,
            source="snapshot",
            as_of=snapshot.as_of_date,
            meta=_opening_meta("snapshot", account.account_id, snapshot.as_of_date),
        )

    return await _running_balance_fallback(session, account, cutover)


async def _latest_snapshot(
    session: AsyncSession, account_id: int, cutover: datetime.date
) -> BalanceSnapshot | None:
    stmt = (
        select(BalanceSnapshot)
        .where(
            BalanceSnapshot.account_id == account_id,
            BalanceSnapshot.as_of_date <= cutover,
            # Snapshot rows have always carried an explicit commodity in the
            # net-worth model. A legacy NULL is unknown, not implicit INR.
            BalanceSnapshot.currency == INR,
        )
        .order_by(BalanceSnapshot.as_of_date.desc(), BalanceSnapshot.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def _running_balance_fallback(
    session: AsyncSession,
    account: LedgerAccount,
    cutover: datetime.date,
) -> OpeningBalance | None:
    """Fallback opening from the latest pre-cutover running balance.

    Only bank transactions carry a meaningful running ``balance``; for cards it
    is usually NULL, so the fallback returns ``None`` and the account opens at
    zero (reported, not fabricated).
    """
    stmt = (
        select(Transaction)
        .where(
            Transaction.account_id == account.account_id,
            Transaction.transaction_date.is_not(None),
            Transaction.transaction_date <= cutover,
            Transaction.balance.is_not(None),
            # Transaction.currency predates its non-NULL default; NULL is the
            # established legacy representation of INR. Explicit foreign rows
            # can never seed an INR opening.
            or_(Transaction.currency == INR, Transaction.currency.is_(None)),
        )
        .order_by(Transaction.transaction_date.desc(), Transaction.id.desc())
        .limit(1)
    )
    txn = (await session.execute(stmt)).scalars().first()
    if txn is None or txn.balance is None:
        return None
    value = Decimal(txn.balance)
    if account.kind == "liability":
        value = -value
    return OpeningBalance(
        account_id=account.account_id,
        account_name=account.name,
        amount=value,
        source="transaction_balance",
        as_of=txn.transaction_date,
        meta=_opening_meta(
            "transaction_balance", account.account_id, txn.transaction_date
        ),
    )


# ---------------------------------------------------------------------------
# Self-transfer pairing
# ---------------------------------------------------------------------------


def _pair_self_transfers(
    txns: list[Transaction],
) -> tuple[dict[str, list[Transaction]], list[Transaction]]:
    """Group ``self_transfer`` rows by reference into candidate pairs.

    Returns ``(groups, leftovers)``: ``groups`` maps a non-empty reference to
    its (>=1) same-reference rows; ``leftovers`` are self-transfer rows with no
    usable reference, which cannot be paired and are reported unmatched.
    """
    groups: dict[str, list[Transaction]] = {}
    leftovers: list[Transaction] = []
    for txn in txns:
        if (txn.category or "").lower() != SELF_TRANSFER_SLUG:
            continue
        ref = (txn.reference_number or "").strip()
        if not ref:
            leftovers.append(txn)
            continue
        groups.setdefault(ref, []).append(txn)
    return groups, leftovers


# ---------------------------------------------------------------------------
# Canonical metadata assembly
# ---------------------------------------------------------------------------


def _pipe(values) -> str:
    """Pipe-separated string of non-None values (``"1|3"``). Empty when none."""
    return "|".join(str(v) for v in values if v is not None)


def _txn_meta(
    txn: Transaction,
    kind: str,
    account_ids: tuple[int | None, ...],
    card_ids: tuple[int | None, ...] = (),
    extra: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Build the canonical ``dashboard_*`` entry-level metadata tuple.

    Every emitted entry carries this closed schema so an operator can drill
    from any ledger line back to its dashboard source rows without parsing
    free text. Values are sanitized (no secrets, raw bodies, or full masks) —
    only non-sensitive fields (ids, slugs, channels, references) appear, and
    :func:`sanitize_meta_value` is the last guard.

    ``dashboard_reference`` is emitted only when a non-empty reference exists.
    ``extra`` lets a caller add entry-specific fields (e.g. card resolution
    status) after the canonical block.
    """
    slug = (txn.category or "").strip().lower() or "unknown"
    ids = tuple(tid for tid in (txn.id,) if tid is not None)
    meta: list[tuple[str, str]] = [
        ("dashboard_txn_ids", "|".join(f"txn-{i}" for i in ids) or "txn-"),
        ("dashboard_kind", kind),
        ("dashboard_category", slug),
        ("dashboard_source", sanitize_meta_value(txn.source) or "unknown"),
        ("dashboard_channel", sanitize_meta_value(txn.channel) or "unknown"),
        ("dashboard_email_type", sanitize_meta_value(txn.email_type) or "unknown"),
        ("dashboard_account_ids", _pipe(account_ids) or "none"),
        ("dashboard_card_ids", _pipe(card_ids) or "none"),
    ]
    ref = sanitize_meta_value(txn.reference_number)
    if ref:
        meta.append(("dashboard_reference", ref))
    meta.extend(extra)
    return tuple(meta)


def _opening_meta(
    ob_source: str, account_id: int, as_of
) -> tuple[tuple[str, str], ...]:
    """Posting-level metadata for an opening-balance posting."""
    return (
        ("dashboard_account_ids", str(account_id)),
        ("dashboard_source", sanitize_meta_value(ob_source) or "unknown"),
        ("dashboard_as_of", str(as_of) if as_of is not None else "unknown"),
    )


# ---------------------------------------------------------------------------
# Entry assembly
# ---------------------------------------------------------------------------


def _build_standard_entry(
    txn: Transaction,
    account: LedgerAccount,
    config: PaisaProjectionConfig,
    backend: str,
    commodity: str,
    *,
    contra_override: str | None = None,
    kind_override: str | None = None,
) -> ProjectedEntry:
    """A two-posting entry: the account ↔ its category contra, in ``commodity``.

    ``contra_override`` lets the caller remap the contra to a non-default
    account (e.g. an investment-funding bank leg remapped to
    :data:`INVESTMENT_EQUITY_OPENING` to avoid double-counting a lot). When
    set, the ``dashboard_kind`` is ``kind_override`` (defaulting to the
    category-derived kind) so the entry is still traceable.
    """
    sign = _account_sign(txn.direction)
    slug = (txn.category or "").strip().lower() or "unknown"
    kind = kind_override or category_kind(slug)
    if contra_override is not None:
        contra = contra_override
    else:
        contra = contra_account(txn.category, config, backend)
    date = txn.transaction_date
    assert date is not None  # caller filters transaction_date.is_not(None)
    card_ids = (txn.card_id,) if txn.card_id is not None else ()
    meta = _txn_meta(
        txn,
        kind,
        account_ids=(account.account_id,),
        card_ids=card_ids,
    )
    return ProjectedEntry(
        date=date,
        payee=txn.counterparty or txn.bank,
        txn_ids=(txn.id,) if txn.id is not None else (),
        postings=(
            LedgerPosting(
                account=account.name,
                amount=Decimal(txn.amount) * sign,
                commodity=commodity,
            ),
            LedgerPosting(
                account=contra, amount=Decimal(txn.amount) * -sign, commodity=commodity
            ),
        ),
        note=txn.note,
        currency=commodity,
        kind=kind,
        meta=meta,
    )


def _build_card_payment_entry(
    txn: Transaction,
    bank_account: LedgerAccount,
    config: PaisaProjectionConfig,
    backend: str,
    commodity: str,
    *,
    resolution: _CardPaymentResolution,
) -> ProjectedEntry:
    """A card-payment bank debit: bank asset ↓, card liability ↓, in ``commodity``.

    **Card resolution** (requirement: never fuzzy match):

    * If ``resolution.card`` is provided (the bank row has an explicit
      ``card_id`` or an exact card mask that maps to a selected card account),
      the payment posts to that specific liability.
    * Otherwise it posts to the generic clearing liability
      (:data:`CARD_PAYMENT_CLEARING`) and the entry carries
      ``dashboard_card_resolution=unresolved`` metadata so the operator knows
      the pairing was not determined.

    Only invoked for an ASSET (bank) account: the caller skips card-side
    ``credit_card_payment`` legs so they are not misposted here.

    The sign follows ``txn.direction``: a debit is the normal payment (bank ↓,
    liability ↓ i.e. positive on the credit-normal liability), a credit is a
    payment reversal that returns money to the bank (bank ↑, liability ↑). A
    fixed outflow sign would mispost every reversal as a second payment.
    """
    sign = _account_sign(txn.direction)
    amount = Decimal(txn.amount)
    date = txn.transaction_date
    assert date is not None  # caller filters transaction_date.is_not(None)
    resolved = resolution.card
    if resolved is not None:
        liability_name = resolved.account.name
        # Only Card.id values belong in dashboard_card_ids. The selected
        # liability's Account.id is independently traceable under
        # dashboard_account_ids, even when the two numeric id spaces collide.
        card_ids = (resolved.card_id,)
        account_ids = (bank_account.account_id, resolved.account.account_id)
    else:
        liability_name = card_clearing_account(config, backend)
        card_ids = ()
        account_ids = (bank_account.account_id,)
    extra = (("dashboard_card_resolution", resolution.status),)
    meta = _txn_meta(
        txn,
        KIND_CARD_PAYMENT,
        account_ids=account_ids,
        card_ids=card_ids,
        extra=extra,
    )
    return ProjectedEntry(
        date=date,
        payee=txn.counterparty or "Card Payment",
        txn_ids=(txn.id,) if txn.id is not None else (),
        postings=(
            LedgerPosting(
                account=liability_name, amount=amount * -sign, commodity=commodity
            ),
            LedgerPosting(
                account=bank_account.name, amount=amount * sign, commodity=commodity
            ),
        ),
        note=txn.note,
        currency=commodity,
        kind=KIND_CARD_PAYMENT,
        meta=meta,
    )


def _build_self_transfer_entry(
    debit: Transaction,
    credit: Transaction,
    accounts: dict[int, LedgerAccount],
    commodity: str,
) -> ProjectedEntry | None:
    """One balanced entry for a matched debit/credit self-transfer pair."""
    debit_acct = accounts.get(debit.account_id) if debit.account_id else None
    credit_acct = accounts.get(credit.account_id) if credit.account_id else None
    if debit_acct is None or credit_acct is None:
        return None
    amount = Decimal(debit.amount)
    debit_date = debit.transaction_date
    credit_date = credit.transaction_date
    assert debit_date is not None and credit_date is not None  # filtered in scope
    ids = tuple(sorted(tid for tid in (debit.id, credit.id) if tid is not None))
    # Build merged metadata from both legs: account ids from both, the debit's
    # source/channel/email_type as the primary identity.
    account_ids = tuple(
        a for a in (debit.account_id, credit.account_id) if a is not None
    )
    card_ids = tuple(c for c in (debit.card_id, credit.card_id) if c is not None)
    meta = _txn_meta(
        debit,
        KIND_SELF_TRANSFER,
        account_ids=account_ids,
        card_ids=card_ids,
    )
    # Override dashboard_txn_ids with the pair's ids (the helper built from
    # the debit's single id).
    meta = tuple(
        (k, "|".join(f"txn-{i}" for i in ids) if k == "dashboard_txn_ids" else v)
        for k, v in meta
    )
    return ProjectedEntry(
        # Earliest observation date is the truer event date (see txn_merge).
        date=min(debit_date, credit_date),
        payee=debit.counterparty or credit.counterparty or "Self Transfer",
        txn_ids=ids,
        postings=(
            # Money leaves the debit account and lands in the credit account.
            LedgerPosting(account=debit_acct.name, amount=-amount, commodity=commodity),
            LedgerPosting(account=credit_acct.name, amount=amount, commodity=commodity),
        ),
        note=debit.reference_number or credit.reference_number,
        currency=commodity,
        kind=KIND_SELF_TRANSFER,
        meta=meta,
    )


def _price_directive_for(
    txn: Transaction, decision: FxDecision
) -> PriceDirective | None:
    """The price directive a priced foreign transaction contributes (or None).

    Dated at the transaction's own date; the projection deduplicates across
    entries so two priced rows on the same date share one directive.
    """
    if (
        decision.currency == INR
        or decision.rate is None
        or txn.transaction_date is None
    ):
        return None
    return PriceDirective(
        date=txn.transaction_date,
        currency=decision.currency,
        rate=decision.rate,
    )


# ---------------------------------------------------------------------------
# CAS portfolio valuation (aggregate only)
# ---------------------------------------------------------------------------
#
# Every CAS portfolio is projected from its authoritative INR ``BalanceSnapshot``
# history and nothing else. There is no accounting-mode selection, no cost-basis
# path, and no lot consumption — the journal carries plain INR balances with no
# cost annotation, no acquisition date and no commodity, so no consumer can
# derive a capital gain or XIRR from them.
#
# Statement-effective semantics. The value source is the one
# :func:`financial_dashboard.services.networth.current_networth` selects, so the
# ledger's CAS component equals the dashboard's by construction. The latest
# snapshot on or before the cutover becomes an opening balance; each later
# statement posts only the *delta* since the previous value; a portfolio falling
# to zero posts the negative of its prior value and clears; an unchanged value
# posts nothing. Because every posting is dated at its own statement date, a
# future statement can only ever affect balances from that date forward — it can
# never rewrite an earlier one.


class _PortfolioValuation(NamedTuple):
    """One CAS portfolio's authoritative INR value history."""

    portfolio_key: str
    #: ``(as_of_date, value)`` ascending, one entry per source snapshot date.
    points: tuple[tuple[datetime.date, Decimal], ...]


async def _load_portfolio_valuations(
    session: AsyncSession, *, as_of: datetime.date | None = None
) -> dict[str, _PortfolioValuation]:
    """Authoritative INR CAS portfolio values, keyed by normalized portfolio.

    Mirrors :func:`financial_dashboard.services.networth.current_networth`'s
    source selection: investment-category INR balance snapshots, latest per
    ``(portfolio_key, date)``. ``CasUpload.grand_total`` and the snapshot value
    are written together by CAS ingestion, so reading the snapshot keeps exact
    parity with native net worth without assuming instrument rows sum to the
    portfolio total.

    Snapshots dated after *as_of* are excluded so a future statement never
    affects an earlier balance.
    """
    rows = (
        (
            await session.execute(
                select(BalanceSnapshot)
                .outerjoin(CasUpload, BalanceSnapshot.cas_upload_id == CasUpload.id)
                .where(
                    BalanceSnapshot.category == SnapshotCategory.investment.value,
                    BalanceSnapshot.currency == INR,
                    BalanceSnapshot.cas_upload_id.is_not(None),
                )
                .order_by(BalanceSnapshot.as_of_date, BalanceSnapshot.id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    # Grouped by the RAW portfolio key, exactly as
    # :func:`networth._source_key` does — never by a normalized form. Native net
    # worth treats two raw keys differing only in case or whitespace as separate
    # sources and SUMS them; normalizing here would merge their series and (for
    # same-date rows) silently drop one, so the ledger would under-report.
    # Legacy/imported rows can carry such variants, so this must mirror native
    # identity even though ingestion uppercases going forward. The private
    # ledger token is still derived from the normalized key, so two raw
    # identities may post to one account — their amounts still sum and match
    # native net worth without exposing the raw key.
    #
    # Last write wins per (raw portfolio, date) exactly as _latest_per_source
    # does: rows arrive ascending by (as_of_date, id), so a later id on the same
    # date supersedes an earlier one.
    per_portfolio: dict[str, dict[datetime.date, Decimal]] = {}
    for row in rows:
        if as_of is not None and row.as_of_date > as_of:
            continue
        key = row.portfolio_key or (
            row.cas_upload.portfolio_key if row.cas_upload is not None else None
        )
        if key is None or not key.strip():
            continue
        per_portfolio.setdefault(key, {})[row.as_of_date] = Decimal(row.value)
    return {
        portfolio: _PortfolioValuation(
            portfolio_key=portfolio,
            points=tuple(sorted(points.items())),
        )
        for portfolio, points in per_portfolio.items()
    }


def _valuation_entries(
    valuation: _PortfolioValuation,
    *,
    cutover: datetime.date,
    account: str,
    equity_account: str,
    reason: str,
) -> list[ProjectedEntry]:
    """Opening + delta postings representing one portfolio's value over time.

    The latest snapshot on or before the cutover is the opening balance (dated
    at the cutover, like every other opening). Each later snapshot posts only
    the change since the previous value, so the account's balance at any date
    equals the CAS statement value in force on that date — including zero, when
    a portfolio is emptied. Zero deltas emit nothing, keeping the journal
    byte-stable when a statement repeats a value.
    """
    opening = Decimal("0.00")
    later: list[tuple[datetime.date, Decimal]] = []
    #: The statement date the opening value was actually observed on. The entry
    #: itself is dated at the cutover (that is when the balance is struck), but
    #: the metadata must retain the real source date or a stale valuation looks
    #: like it was observed at cutover and audit provenance is lost.
    opening_as_of: datetime.date | None = None
    for as_of_date, value in valuation.points:
        if as_of_date <= cutover:
            opening = value
            opening_as_of = as_of_date
        else:
            later.append((as_of_date, value))

    entries: list[ProjectedEntry] = []
    running = Decimal("0.00")

    def _meta(as_of_date: datetime.date, kind: str) -> tuple[tuple[str, str], ...]:
        return (
            ("dashboard_kind", KIND_VALUATION),
            ("dashboard_valuation_kind", kind),
            ("dashboard_portfolio_token", sanitize_meta_value(account.split(":")[-1])),
            ("dashboard_as_of", as_of_date.isoformat()),
            ("dashboard_valuation_reason", sanitize_meta_value(reason) or "unknown"),
            # Stated so no consumer reads the balance as an acquisition cost.
            ("dashboard_cost_basis_available", "false"),
        )

    if opening:
        running = opening
        entries.append(
            ProjectedEntry(
                date=cutover,
                payee="CAS Portfolio Valuation",
                txn_ids=(),
                postings=(
                    LedgerPosting(account=account, amount=opening, commodity=INR),
                    LedgerPosting(
                        account=equity_account, amount=-opening, commodity=INR
                    ),
                ),
                note=None,
                currency=INR,
                kind=KIND_VALUATION,
                # Entry is dated at the cutover; the metadata keeps the real
                # statement date the value was observed on.
                meta=_meta(opening_as_of or cutover, "opening"),
            )
        )

    for as_of_date, value in later:
        delta = value - running
        running = value
        if delta == 0:
            continue
        entries.append(
            ProjectedEntry(
                date=as_of_date,
                payee="CAS Portfolio Revaluation",
                txn_ids=(),
                postings=(
                    LedgerPosting(account=account, amount=delta, commodity=INR),
                    LedgerPosting(account=equity_account, amount=-delta, commodity=INR),
                ),
                note=None,
                currency=INR,
                kind=KIND_VALUATION,
                meta=_meta(as_of_date, "revaluation"),
            )
        )
    return entries


def _valuation_is_active(value: CurrentValuation) -> bool:
    """Whether a latest source identity reports a positive current position."""
    return bool(
        (value.quantity is not None and value.quantity > 0)
        or (value.value is not None and value.value > 0)
    )


def _valuation_source_label(value: CurrentValuation) -> str:
    """Stable folio/demat identity for diagnostics (never a holding posting)."""
    return (
        f"{value.portfolio_key}/{value.scope}/{value.source_ref}/"
        f"{value.instrument_id}#{value.occurrence}"
    )


async def _load_net_worth_scope_sources(
    session: AsyncSession,
) -> _NetWorthScopeSources:
    """Name every active native net-worth source outside Account selection."""
    uploads = (
        (
            await session.execute(
                select(CasUpload).order_by(
                    CasUpload.statement_date.desc(), CasUpload.id.desc()
                )
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, CasUpload] = {}
    for upload in uploads:
        latest.setdefault(upload.portfolio_key.strip().upper(), upload)
    cas_labels = tuple(
        sorted(
            (
                f"{key} ({upload.investor_name.strip()})"
                if upload.investor_name and upload.investor_name.strip()
                else key
            )
            for key, upload in latest.items()
        )
    )

    manual_items = (
        (
            await session.execute(
                select(ManualItem)
                .where(ManualItem.active.is_not(False))
                .order_by(ManualItem.kind, ManualItem.name, ManualItem.id)
            )
        )
        .scalars()
        .all()
    )
    asset_labels = tuple(
        f"{item.id}: {item.name}" for item in manual_items if item.kind == "asset"
    )
    liability_labels = tuple(
        f"{item.id}: {item.name}" for item in manual_items if item.kind != "asset"
    )
    return _NetWorthScopeSources(
        cas_portfolio_labels=cas_labels,
        cas_portfolio_keys=frozenset(latest),
        manual_asset_labels=asset_labels,
        manual_liability_labels=liability_labels,
    )


# ---------------------------------------------------------------------------
# Card payment resolution (exact card_id / exact mask only — never fuzzy)
# ---------------------------------------------------------------------------


async def _load_card_resolution_maps(
    session: AsyncSession,
    accounts_by_id: dict[int, LedgerAccount],
) -> _CardResolutionMaps:
    """Build selected-card id and globally unique exact-mask lookups.

    Only card-type (liability) accounts that are in the selected set are
    considered. A bank-side card payment resolves to a specific liability ONLY
    when the row carries an explicit ``card_id`` whose Card belongs to a
    selected account, OR an exact ``card_mask`` match. **Never** fuzzy: a
    partial or near-miss mask does not resolve. A mask shared by two selected
    Card rows is retained only in ``ambiguous_masks`` and never points to a
    last-query-order winner. Card primary keys are globally unique, so an
    explicit selected ``card_id`` remains authoritative.
    """
    liability_accounts = {
        aid: acct for aid, acct in accounts_by_id.items() if acct.kind == "liability"
    }
    if not liability_accounts:
        return _CardResolutionMaps({}, {}, frozenset())
    card_rows = (
        (
            await session.execute(
                select(Card).where(Card.account_id.in_(set(liability_accounts.keys())))
            )
        )
        .scalars()
        .all()
    )
    by_card_id: dict[int, _ResolvedCard] = {}
    mask_candidates: dict[str, list[_ResolvedCard]] = {}
    for card in card_rows:
        acct = liability_accounts.get(card.account_id)
        if acct is None:
            continue
        if card.id is not None:
            resolved = _ResolvedCard(card_id=card.id, account=acct)
            by_card_id[card.id] = resolved
        else:
            continue
        if card.card_mask:
            mask_candidates.setdefault(card.card_mask.strip(), []).append(resolved)
    by_unique_mask = {
        mask: candidates[0]
        for mask, candidates in mask_candidates.items()
        if mask and len(candidates) == 1
    }
    ambiguous_masks = frozenset(
        mask
        for mask, candidates in mask_candidates.items()
        if mask and len(candidates) > 1
    )
    return _CardResolutionMaps(by_card_id, by_unique_mask, ambiguous_masks)


def _resolve_card_for_payment(
    txn: Transaction,
    maps: _CardResolutionMaps,
) -> _CardPaymentResolution:
    """Resolve a card-payment bank leg to a specific selected liability.

    Exact-match only: ``txn.card_id`` → ``by_card_id``, or ``txn.card_mask``
    → the unique-mask map. Shared exact masks return ``ambiguous_mask`` and
    generic clearing; no query-order winner can leak through. No exact match
    returns ``unresolved`` — never a fuzzy/text-similarity guess.
    """
    if txn.card_id is not None:
        card = maps.by_card_id.get(txn.card_id)
        if card is not None:
            return _CardPaymentResolution(card, "resolved")
    mask = (txn.card_mask or "").strip()
    if mask:
        if mask in maps.ambiguous_masks:
            return _CardPaymentResolution(None, "ambiguous_mask")
        card = maps.by_unique_mask.get(mask)
        if card is not None:
            return _CardPaymentResolution(card, "resolved")
    return _CardPaymentResolution(None, "unresolved")


# ---------------------------------------------------------------------------
# Top-level projection
# ---------------------------------------------------------------------------


async def project(
    session: AsyncSession, config: PaisaProjectionConfig
) -> ProjectionReport:
    """Project the selected accounts' post-cutover activity into a journal.

    Read-only. Raises :class:`ProjectionError` if a cutover date is not
    configured — projection without a cutover would emit an opening-less
    journal whose running balances are meaningless.
    """
    if config.cutover_date is None:
        raise ProjectionError(
            "paisa.project_since (cutover date) is required for projection."
        )

    cutover = config.cutover_date
    selected = set(config.selected_account_ids)
    backend = config.ledger_cli
    scope_sources = await _load_net_worth_scope_sources(session)
    current_valuations = await get_current_valuations(session)

    # ---- load & resolve accounts ------------------------------------------
    account_rows = (
        (await session.execute(select(Account).where(Account.id.in_(selected))))
        .scalars()
        .all()
    )
    accounts_by_id: dict[int, LedgerAccount] = {}
    missing_ids = sorted(selected - {a.id for a in account_rows})
    skipped: list[SkippedRow] = [
        SkippedRow(txn_id=None, reason="unknown_account", detail=f"account {aid}")
        for aid in missing_ids
    ]
    for acct in account_rows:
        accounts_by_id[acct.id] = resolve_account(
            acct, config.account_mappings, backend
        )

    # ---- card resolution maps (exact card_id / mask → liability) ------------
    card_maps = await _load_card_resolution_maps(session, accounts_by_id)

    # ---- openings ---------------------------------------------------------
    openings_list: list[OpeningBalance] = []
    for acct in sorted(accounts_by_id.values(), key=lambda a: a.account_id):
        opening = await _opening_for_account(session, acct, cutover)
        if opening is not None:
            openings_list.append(opening)

    # ---- transactions -----------------------------------------------------
    txns = (
        (
            await session.execute(
                select(Transaction)
                .where(
                    Transaction.account_id.in_(selected),
                    Transaction.transaction_date.is_not(None),
                    Transaction.transaction_date > cutover,
                )
                .order_by(Transaction.transaction_date, Transaction.id)
            )
        )
        .scalars()
        .all()
    )

    entries: list[ProjectedEntry] = []
    non_inr = 0
    missing_fx = 0
    unmatched = 0
    unknown = 0
    card_payments = 0
    card_side_payments = 0
    card_payments_resolved = 0
    card_payments_unresolved = 0
    card_payments_ambiguous_mask = 0
    imprecise = 0
    self_pairs = 0
    investment_unresolved_purchases = 0
    investment_unresolved_redemptions = 0
    source_currencies: set[str] = set()
    prices: list[PriceDirective] = []

    # FX classification happens BEFORE self-transfer pairing: a non-INR row
    # under the skip policy is never emitted (and never paired), so a
    # self-transfer leg in another currency cannot drag a foreign amount into an
    # INR-labelled posting. Under the priced policy a foreign leg with a rate is
    # eligible; one without is skipped as missing_fx_rate.
    decisions: dict[int, FxDecision] = {}
    eligible: list[Transaction] = []
    for txn in txns:
        decision = _decide_fx(txn, config)
        if decision.skip_reason == "invalid_currency":
            # A non-empty but unrepresentable currency (control chars, digit-led,
            # punctuation-only) is skipped with a clear diagnostic so it can
            # never reach a posting amount or a price directive in any backend.
            skipped.append(
                SkippedRow(
                    txn_id=txn.id,
                    reason="invalid_currency",
                    detail=f"unrepresentable currency {txn.currency!r}",
                )
            )
            continue
        if decision.skip_reason == "non_inr":
            non_inr += 1
            skipped.append(
                SkippedRow(
                    txn_id=txn.id,
                    reason="non_inr",
                    detail=f"currency {decision.currency!r}",
                )
            )
            continue
        if decision.skip_reason == "missing_fx_rate":
            missing_fx += 1
            skipped.append(
                SkippedRow(
                    txn_id=txn.id,
                    reason="missing_fx_rate",
                    detail=(
                        f"no paisa.fx_rates entry for {decision.currency!r} "
                        f"on/before {txn.transaction_date}"
                    ),
                )
            )
            continue
        eligible.append(txn)
        if decision.currency != INR:
            source_currencies.add(decision.currency)
        if txn.id is not None:
            decisions[txn.id] = decision

    # Pre-collect self-transfers and remove them from the linear pass; they are
    # emitted as paired entries below, never as standalone rows.
    st_groups, st_leftovers = _pair_self_transfers(eligible)
    st_seen_ids: set[int] = set()
    for txn in st_leftovers:
        unmatched += 1
        skipped.append(
            SkippedRow(
                txn_id=txn.id,
                reason="unmatched_self_transfer",
                detail="self_transfer with no reference number",
            )
        )
        if txn.id is not None:
            st_seen_ids.add(txn.id)

    for ref, group in sorted(st_groups.items()):
        debits = [t for t in group if t.direction == "debit"]
        credits = [t for t in group if t.direction == "credit"]
        if len(debits) == 1 and len(credits) == 1:
            debit, credit = debits[0], credits[0]
            debit_dec = decisions.get(debit.id)
            credit_dec = decisions.get(credit.id)
            debit_ccy = debit_dec.currency if debit_dec is not None else None
            credit_ccy = credit_dec.currency if credit_dec is not None else None
            if Decimal(debit.amount) != Decimal(credit.amount):
                # A genuine same-ref pair should match in magnitude; otherwise
                # the reference is shared by distinct events and we refuse to
                # collapse them.
                unmatched += len(group)
                for t in group:
                    skipped.append(
                        SkippedRow(
                            txn_id=t.id,
                            reason="unmatched_self_transfer",
                            detail=f"reference {ref}: amount mismatch",
                        )
                    )
                    if t.id is not None:
                        st_seen_ids.add(t.id)
                continue
            if debit_ccy != credit_ccy or debit_ccy is None:
                # A clean pair must share one currency; a cross-currency pair
                # would need an FX conversion we deliberately do not perform.
                unmatched += len(group)
                for t in group:
                    skipped.append(
                        SkippedRow(
                            txn_id=t.id,
                            reason="unmatched_self_transfer",
                            detail=(
                                f"reference {ref}: currency mismatch "
                                f"({debit_ccy!r} vs {credit_ccy!r})"
                            ),
                        )
                    )
                    if t.id is not None:
                        st_seen_ids.add(t.id)
                continue
            entry = _build_self_transfer_entry(debit, credit, accounts_by_id, debit_ccy)
            if entry is None:
                unmatched += len(group)
                for t in group:
                    skipped.append(
                        SkippedRow(
                            txn_id=t.id,
                            reason="unmatched_self_transfer",
                            detail=f"reference {ref}: account not in scope",
                        )
                    )
                    if t.id is not None:
                        st_seen_ids.add(t.id)
                continue
            entries.append(entry)
            self_pairs += 1
            for t in group:
                if t.id is not None:
                    st_seen_ids.add(t.id)
                price = (
                    _price_directive_for(t, decisions[t.id])
                    if t.id in decisions
                    else None
                )
                if price is not None:
                    prices.append(price)
        else:
            # Not a clean 1+1 pair — refuse to guess which legs go together.
            unmatched += len(group)
            for t in group:
                skipped.append(
                    SkippedRow(
                        txn_id=t.id,
                        reason="unmatched_self_transfer",
                        detail=(
                            f"reference {ref}: {len(debits)} debit(s), "
                            f"{len(credits)} credit(s)"
                        ),
                    )
                )
                if t.id is not None:
                    st_seen_ids.add(t.id)

    # ---- CAS investments: aggregate valuation only --------------------------
    # Projection deliberately does NOT consume InvestmentLot rows. Those rows
    # remain a first-class ingestion fact (the model, normalization, backfill and
    # dashboard views are untouched), but the *journal* is built solely from each
    # portfolio's authoritative BalanceSnapshot history.
    #
    # Why: reconciling per-lot cost basis against per-statement CAS aggregates is
    # not determined by this data. A bank row does not name the portfolio it
    # funded, and a CAS value change is purchases minus redemptions PLUS market
    # movement, so the cash-flow component is unrecoverable. Every attempt to
    # bridge that gap either double counted, deleted assets, drove an asset
    # balance negative, or let a later statement rewrite an earlier balance.
    # See the preserved design note for what a correct cost-basis feature needs.
    #: Projection-relevant policy diagnostics ONLY. Lot-normalization reasons
    #: (non-MF, missing cost facts, ...) are deliberately NOT surfaced here: the
    #: authoritative aggregate *includes* those holdings' value, so reporting
    #: them as "excluded" would imply value was omitted when it was not. The
    #: core investment service still exposes them on its own dashboard surface.
    investment_excluded: tuple[str, ...] = ()
    #: Whether CAS aggregates are being projected at all. Gates both the
    #: unresolved-leg policy and the valuation emission below.
    cas_portfolios_projected = bool(
        config.project_investments and scope_sources.cas_portfolio_keys
    )
    portfolio_values: dict[str, _PortfolioValuation] = {}
    # Read once: the valuation account names derive a non-reversible portfolio
    # token from it. Projection only READS the secret; the migration creates it.
    secret = get_setting(PORTFOLIO_TOKEN_SECRET_KEY)
    if cas_portfolios_projected:
        portfolio_values = await _load_portfolio_valuations(session)

    # Linear pass over the remaining (eligible, non-self-transfer) txns.
    for txn in eligible:
        if txn.id is not None and txn.id in st_seen_ids:
            continue
        if (txn.category or "").lower() == SELF_TRANSFER_SLUG:
            # Stray self-transfer leg that slipped past grouping — never emit
            # standalone, surface it for manual resolution instead.
            unmatched += 1
            skipped.append(
                SkippedRow(
                    txn_id=txn.id,
                    reason="unmatched_self_transfer",
                    detail="self_transfer leg not in a clean pair",
                )
            )
            continue

        account = accounts_by_id.get(txn.account_id) if txn.account_id else None
        if account is None:
            # account_id is NULL or outside the selected set. The query filters
            # account_id IN selected, so this is a NULL account_id row — skip
            # rather than crash.
            skipped.append(
                SkippedRow(
                    txn_id=txn.id,
                    reason="orphan",
                    detail="transaction has no selected account",
                )
            )
            continue

        category = (txn.category or "").lower()
        decision = decisions.get(txn.id) if txn.id is not None else None
        commodity = decision.commodity if decision is not None else INR

        if category == CREDIT_CARD_PAYMENT_SLUG:
            if account.kind == "liability":
                # A card-side ``credit_card_payment`` leg is the card being
                # paid down. The bank-side leg is the authoritative payment
                # event and is what we project; emitting the card-side leg too
                # would double-count, and we cannot fabricate the bank it came
                # from. Surface it rather than mispost.
                card_side_payments += 1
                skipped.append(
                    SkippedRow(
                        txn_id=txn.id,
                        reason="card_side_payment",
                        detail="credit_card_payment on a card account; emit the bank leg",
                    )
                )
                continue
            # Card resolution: exact card_id or exact mask → specific liability;
            # otherwise generic clearing with dashboard_card_resolution=unresolved.
            card_resolution = _resolve_card_for_payment(txn, card_maps)
            entries.append(
                _build_card_payment_entry(
                    txn,
                    account,
                    config,
                    backend,
                    commodity,
                    resolution=card_resolution,
                )
            )
            card_payments += 1
            if card_resolution.card is not None:
                card_payments_resolved += 1
            else:
                card_payments_unresolved += 1
                if card_resolution.status == "ambiguous_mask":
                    card_payments_ambiguous_mask += 1
        else:
            if category in ("", "unknown"):
                unknown += 1
            if category in IMPRECISE_CATEGORY_SLUGS:
                imprecise += 1
            # Every bank investment leg is UNRESOLVED by policy. No funding
            # matching, no remapping, no netting against the CAS aggregate: the
            # source data cannot say which portfolio a bank row funded, so any
            # inferred link would be fabricated. The asset stays where it
            # honestly belongs and the ambiguity is reported instead.
            #
            # An explicit operator ``category_mappings`` entry outranks this (and
            # every other generated policy) — ``contra_account()`` documents that
            # contract, and silently bypassing a configured account would leave
            # the operator's holdings account understated with no signal.
            contra_override = None
            kind_override = None
            explicit_mapping = category in config.category_mappings
            if (
                category in INVESTMENT_CATEGORY_SLUGS
                and not explicit_mapping
                and cas_portfolios_projected
            ):
                if txn.direction == "debit":
                    # A purchase: the money really did move into investments, so
                    # the asset stays in Assets:Investments:Unallocated. The CAS
                    # aggregate may also contain it, which is a possible overlap
                    # an operator must interpret — reported, never netted.
                    investment_unresolved_purchases += 1
                else:
                    # A redemption with no projected lot to retire. It must NOT
                    # post a negative amount to Assets:Investments:Unallocated:
                    # that account only holds this projection's own unmatched
                    # purchases, while a redemption usually disposes of holdings
                    # acquired before the cutover that were never projected.
                    # Netting it there drives an asset negative — meaningless
                    # accounting, and a fatal Paisa "Negative Balance" diagnosis.
                    contra_override = normalize_policy_account(
                        REPAYMENT_CLEARING_ACCOUNT,
                        backend=backend,
                        label="investment_redemption_unallocated",
                    )
                    kind_override = KIND_INVESTMENT
                    investment_unresolved_redemptions += 1
            entries.append(
                _build_standard_entry(
                    txn,
                    account,
                    config,
                    backend,
                    commodity,
                    contra_override=contra_override,
                    kind_override=kind_override,
                )
            )

        # Record the price directive for a priced foreign standard/card entry.
        if decision is not None and (price := _price_directive_for(txn, decision)):
            prices.append(price)

    # Stable ordering: opening entry is rendered first by the renderer; here we
    # keep entries sorted by (date, first txn id) so re-runs are byte-identical.
    entries.sort(key=lambda e: (e.date, e.txn_ids[0] if e.txn_ids else 0))

    declared = sorted({a.name for a in accounts_by_id.values()})

    # ---- CAS aggregate valuation entries ----------------------------------
    # One balanced opening + forward deltas per portfolio, dated at each
    # statement. No lots, no market prices, no netting against bank legs.
    valuation_entries: list[ProjectedEntry] = []
    valuation_portfolios: list[str] = []
    valuation_total = Decimal("0.00")
    valuation_unrepresented: list[str] = []
    if cas_portfolios_projected:
        equity_account = normalize_policy_account(
            EQUITY_REVALUATION, backend=backend, label="investment_revaluation"
        )
        # Iterate RAW source identities (matching native net worth's grouping),
        # but derive the account/report identity from the normalized key. Two
        # raw keys that normalize alike post to the same private account, where
        # their independent series sum — which is exactly what native does.
        represented_normalized: set[str] = set()
        for raw_portfolio, series in sorted(portfolio_values.items()):
            if not series.points:
                continue
            normalized = normalize_portfolio_key(raw_portfolio)
            token = portfolio_token(raw_portfolio, secret)
            account = normalize_policy_account(
                investment_valuation_account(token),
                backend=backend,
                label="investment_valuation",
            )
            portfolio_entries = _valuation_entries(
                series,
                cutover=cutover,
                account=account,
                equity_account=equity_account,
                reason="cas_aggregate_valuation",
            )
            # A portfolio whose whole history is zero needs no posting but IS
            # represented: its true value (nothing) is in the ledger exactly.
            represented_normalized.add(normalized)
            if not portfolio_entries:
                continue
            valuation_entries.extend(portfolio_entries)
            valuation_total += series.points[-1][1]
        valuation_portfolios.extend(sorted(represented_normalized))
        # A CAS portfolio with no authoritative INR snapshot has no value to
        # project. Diagnosed, never invented.
        valuation_unrepresented.extend(
            sorted(scope_sources.cas_portfolio_keys - represented_normalized)
        )

    if valuation_entries:
        # Valuation entries carry no txn_ids, so portfolios sharing a statement
        # date would tie at (date, 0); break it on the deterministic account
        # name so the rendered file stays byte-identical across runs.
        entries.extend(valuation_entries)
        entries.sort(
            key=lambda e: (
                e.date,
                e.txn_ids[0] if e.txn_ids else 0,
                e.postings[0].account if not e.txn_ids and e.postings else "",
            )
        )
        investment_excluded = tuple(
            dict.fromkeys((*investment_excluded, "valuation_only_no_cost_basis"))
        )
    if valuation_unrepresented:
        investment_excluded = tuple(
            dict.fromkeys((*investment_excluded, "portfolio_value_unavailable"))
        )

    # Deduplicate price directives by (date, currency): the rate is a pure
    # function of (currency, date) from the configured map, so this collapses
    # same-day same-currency rows into one directive per backend file.
    # Only FX directives reach here — CAS market prices are never emitted.
    deduped_prices = _dedupe_prices(prices)
    doc = LedgerDocument(
        cutover_date=cutover,
        openings=tuple(openings_list),
        entries=tuple(entries),
        accounts_declared=tuple(declared),
        price_directives=deduped_prices,
    )
    journal = render_document_for_backend(doc, backend)

    foreign_count = sum(1 for e in entries if e.currency != INR)
    kind_counts: dict[str, int] = {}
    for e in entries:
        kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1

    represented = set(valuation_portfolios)
    if not scope_sources.cas_portfolio_keys:
        cas_investment_scope = "none"
        cas_investment_coverage = "none"
    elif not config.project_investments:
        cas_investment_scope = "excluded"
        cas_investment_coverage = "excluded"
    else:
        # Value is represented when every CAS portfolio has an authoritative
        # aggregate in the journal. Coverage is ALWAYS valuation_only: this
        # projection never claims cost basis, gains or XIRR for CAS.
        cas_investment_scope = (
            "included" if scope_sources.cas_portfolio_keys <= represented else "partial"
        )
        cas_investment_coverage = "valuation_only" if represented else "none"

    # Sources present is a different claim from total exact. Every bank
    # investment leg is unresolved by policy, so any of them means the
    # investment total may not match the dashboard's snapshot-based value.
    net_worth_sources_complete = (
        not scope_sources.manual_asset_labels
        and not scope_sources.manual_liability_labels
        and cas_investment_scope in {"none", "included"}
    )
    net_worth_scope_complete = (
        net_worth_sources_complete
        and investment_unresolved_purchases == 0
        and investment_unresolved_redemptions == 0
    )
    return ProjectionReport(
        journal=journal,
        document=doc,
        entries=tuple(entries),
        openings=tuple(openings_list),
        emitted_count=len(entries),
        self_transfer_pairs=self_pairs,
        card_payments=card_payments,
        card_side_payments=card_side_payments,
        non_inr_count=non_inr,
        unmatched_count=unmatched,
        unknown_count=unknown,
        skipped=tuple(skipped),
        cutover_date=cutover,
        account_ids=tuple(sorted(selected)),
        projected_foreign_count=foreign_count,
        missing_fx_rate_count=missing_fx,
        source_currencies=tuple(sorted(source_currencies)),
        # Lot-related fields stay for backward compatibility but are ALWAYS
        # empty/zero: this projection never consumes InvestmentLot rows.
        investment_lot_count=0,
        investment_excluded=investment_excluded,
        investment_disposal_unresolved=(),
        imprecise_count=imprecise,
        card_payments_resolved=card_payments_resolved,
        card_payments_unresolved=card_payments_unresolved,
        card_payments_ambiguous_mask=card_payments_ambiguous_mask,
        investment_funding_remapped=0,
        investment_funding_unresolved=(),
        kind_counts=kind_counts,
        investment_current_valuation_count=len(current_valuations),
        investment_market_price_count=0,
        investment_market_price_conflicts=(),
        investment_value_only_count=sum(
            _valuation_is_active(value) for value in current_valuations
        ),
        investment_quantity_mismatch_count=0,
        investment_missing_market_price_count=0,
        investment_valuation_sources=tuple(
            _valuation_source_label(value) for value in current_valuations
        ),
        cas_portfolio_count=len(scope_sources.cas_portfolio_labels),
        cas_portfolio_labels=scope_sources.cas_portfolio_labels,
        cas_investment_scope=cas_investment_scope,
        cas_investment_coverage=cas_investment_coverage,
        investment_cost_basis_portfolios=(),
        investment_valuation_portfolios=tuple(sorted(valuation_portfolios)),
        investment_valuation_entry_count=len(valuation_entries),
        investment_valuation_total=valuation_total,
        investment_valuation_unrepresented=tuple(sorted(valuation_unrepresented)),
        investment_unresolved_purchases=investment_unresolved_purchases,
        investment_unresolved_redemptions=investment_unresolved_redemptions,
        manual_asset_count=len(scope_sources.manual_asset_labels),
        manual_asset_labels=scope_sources.manual_asset_labels,
        manual_liability_count=len(scope_sources.manual_liability_labels),
        manual_liability_labels=scope_sources.manual_liability_labels,
        net_worth_scope_complete=net_worth_scope_complete,
        net_worth_sources_complete=net_worth_sources_complete,
    )


def _dedupe_prices(prices: list[PriceDirective]) -> tuple[PriceDirective, ...]:
    """Deduplicate by (date, currency), preserving a deterministic sort.

    The rate is deterministic per (currency, date) from the configured map, so
    two directives with the same key carry the same rate; a stable sort makes
    the rendered output byte-identical across runs.
    """
    seen: dict[tuple[datetime.date, str], PriceDirective] = {}
    for price in prices:
        key = (price.date, price.currency)
        if key not in seen:
            seen[key] = price
    return tuple(sorted(seen.values(), key=lambda p: (p.date, p.currency)))


def _empty_report(cutover: datetime.date, backend: str) -> ProjectionReport:
    doc = LedgerDocument(
        cutover_date=cutover, openings=(), entries=(), accounts_declared=()
    )
    return ProjectionReport(
        journal=render_document_for_backend(doc, backend),
        document=doc,
        entries=(),
        openings=(),
        emitted_count=0,
        self_transfer_pairs=0,
        card_payments=0,
        card_side_payments=0,
        non_inr_count=0,
        unmatched_count=0,
        unknown_count=0,
        skipped=(),
        cutover_date=cutover,
        account_ids=(),
        projected_foreign_count=0,
        missing_fx_rate_count=0,
        source_currencies=(),
    )
