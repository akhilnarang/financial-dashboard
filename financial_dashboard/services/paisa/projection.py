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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Card,
    InvestmentLot,
    Transaction,
)
from financial_dashboard.services.cashflow.scope import (
    CARD_ACCOUNT_TYPES,
)
from financial_dashboard.services.categorization.slugs import (
    CREDIT_CARD_ACCOUNT_TYPE,
    REPAYMENT_SLUG,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.renderers import (
    normalize_default_account,
    render_document as render_document_for_backend,
    validate_account_name,
)
from financial_dashboard.services.paisa.renderers.base import (
    CARD_PAYMENT_CLEARING,
    INR,
    INVESTMENT_EQUITY_OPENING,
    INVESTMENT_UNALLOCATED_ACCOUNT,
    InvalidAccountName,
    InvestmentLotEntry,
    LedgerAccount,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    PriceDirective,
    ProjectedEntry,
    REPAYMENT_CLEARING_ACCOUNT,
    sanitize_commodity,
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

SELF_TRANSFER_SLUG = "self_transfer"
CREDIT_CARD_PAYMENT_SLUG = "credit_card_payment"

INCOME_CATEGORY_SLUGS = frozenset({"salary", "interest", "other_income"})
CONTRA_EXPENSE_CATEGORY_SLUGS = frozenset({"refund", "cashback_rewards"})
INVESTMENT_CATEGORY_SLUGS = frozenset({"investment", "investment_redemption"})
#: Expense slugs whose contra is imprecise (no principal/cash account in the
#: source row). They post to a conservative Expenses clearing and the report
#: carries an ``imprecise`` diagnostic rather than fabricating an account.
IMPRECISE_CATEGORY_SLUGS = frozenset({"emi_loan", "cash_withdrawal"})

# Closed dashboard_kind taxonomy. Every emitted entry carries exactly one.
KIND_EXPENSE = "expense"
KIND_INCOME = "income"
KIND_CONTRA_EXPENSE = "contra_expense"
KIND_INVESTMENT = "investment"
KIND_REPAYMENT = "repayment"
KIND_SELF_TRANSFER = "self_transfer"
KIND_CARD_PAYMENT = "card_payment"
KIND_OPENING = "opening"
KIND_LOT = "investment_lot"
KIND_UNKNOWN = "unknown"

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


class ProjectionError(Exception):
    """Raised when projection cannot proceed (e.g. no cutover date configured)."""


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
    #: Investment-lot projection diagnostics. ``investment_lot_count`` is the
    #: number of complete lots emitted (zero when ``project_investments`` is
    #: off); ``investment_excluded`` is the deduplicated set of stable reason
    #: labels for CAS facts that could not become a lot. Lots are read-only
    #: here — projection never writes a core row.
    investment_lot_count: int = 0
    investment_excluded: tuple[str, ...] = ()
    #: Instrument ids whose complete acquisition lots were *suppressed* because
    #: the preserved CAS facts contain a disposal/redemption that cannot be
    #: truthfully allocated to lots (so projecting the gross acquisitions would
    #: overstate holdings). The matching ``disposal_history_unresolved`` label
    #: is also added to ``investment_excluded``. Conservative by design: the
    #: default lot projection never overstates holdings.
    investment_disposal_unresolved: tuple[str, ...] = ()
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
    #: Investment-funding double-count prevention: bank investment legs whose
    #: contra was remapped to :data:`INVESTMENT_EQUITY_OPENING` because a
    #: complete lot provably captured the holding. ``investment_funding_unresolved``
    #: lists instruments whose lot was suppressed because the funding link was
    #: potential but not provably deterministic.
    investment_funding_remapped: int = 0
    investment_funding_unresolved: tuple[str, ...] = ()
    #: ``kind_counts`` is the cardinality of each ``dashboard_kind`` among
    #: emitted entries (excluding openings/lots which are structurally
    #: separate). Used by closed-population tests and operator diagnostics.
    kind_counts: dict[str, int] = {}


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


# ---------------------------------------------------------------------------
# Account resolution
# ---------------------------------------------------------------------------


def _account_kind(account_type: str | None) -> str:
    """Map a dashboard account type to a ledger account kind.

    A credit-card account is a liability (it holds what you owe); everything
    else the linker recognizes (bank_account, debit_card) is an asset. An
    unknown type is treated as an asset — same as the cashflow scope's fallback
    — and surfaced via the projection's account list so the operator notices.
    """
    if account_type == CREDIT_CARD_ACCOUNT_TYPE or account_type in CARD_ACCOUNT_TYPES:
        return "liability"
    return "asset"


def _default_account_name(account: Account, kind: str) -> str:
    """Deterministic default ledger name for an account.

    Banks land under ``Assets:Bank:<bank>:<label>``; cards under
    ``Liabilities:Card:<bank>:<label>``. Operator overrides (via
    ``account_mappings``) take precedence and are applied by the caller.
    """
    bank = _title_segment(account.bank)
    label = _title_segment(account.label)
    if kind == "liability":
        return f"Liabilities:Card:{bank}:{label}"
    return f"Assets:Bank:{bank}:{label}"


def _title_segment(value: str | None) -> str:
    """Render a free-form string as a ledger-safe account path segment.

    Title-cases words and drops every character ledger would misread as
    structure (``:``) or whitespace noise. Underscores become spaces first so
    ``savings_account`` reads as ``Savings Account``.
    """
    text = (value or "").replace("_", " ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(part.capitalize() for part in cleaned.split())


def _finalize_name(raw: str, *, is_default: bool, backend: str, label: str) -> str:
    """Resolve a projection account name for the configured backend.

    Defaults are normalized (so a beancount target gets PascalCase'd segments);
    operator overrides are validated strictly and never silently rewritten — an
    incompatible override surfaces as a :class:`ProjectionError` rather than a
    malformed file.
    """
    name = normalize_default_account(raw, backend) if is_default else raw
    try:
        return validate_account_name(name, backend)
    except InvalidAccountName as exc:
        raise ProjectionError(
            f"{label}: invalid ledger name for {backend!r}: {exc}"
        ) from exc


def _resolve_account(
    account: Account, mappings: dict[str, str], backend: str
) -> LedgerAccount:
    kind = _account_kind(account.type)
    override = mappings.get(str(account.id))
    name = _finalize_name(
        override if override is not None else _default_account_name(account, kind),
        is_default=override is None,
        backend=backend,
        label=f"account {account.id}",
    )
    return LedgerAccount(account_id=account.id, name=name, kind=kind)


# ---------------------------------------------------------------------------
# Contra (category) resolution — dashboard taxonomy semantics
# ---------------------------------------------------------------------------


def _category_kind(slug: str) -> str:
    """Classify a category slug into a closed ``dashboard_kind`` value.

    The kind is the *accounting meaning* of the entry, independent of the
    transaction's direction. It drives the contra-root choice and the
    ``dashboard_kind`` metadata tag. Direction only affects the sign (a
    credit on an expense slug is a reversal that nets, not a different kind).
    """
    if slug in INCOME_CATEGORY_SLUGS:
        return KIND_INCOME
    if slug in CONTRA_EXPENSE_CATEGORY_SLUGS:
        return KIND_CONTRA_EXPENSE
    if slug in INVESTMENT_CATEGORY_SLUGS:
        return KIND_INVESTMENT
    if slug == REPAYMENT_SLUG:
        return KIND_REPAYMENT
    if slug in ("", "unknown", "misc"):
        return KIND_UNKNOWN
    return KIND_EXPENSE


def _contra_account(
    category: str | None,
    direction: str,
    config: PaisaProjectionConfig,
    backend: str,
) -> str:
    """Resolve the contra account a category posts against.

    The contra is rooted by **category semantics**, not direction:

    * income slugs → ``Income:<Title>`` (always)
    * refund/cashback → ``Expenses:<Title>`` (contra-expense: negative expense
      on credit so the refund nets against the original spend)
    * investment/redemption → :data:`INVESTMENT_UNALLOCATED_ACCOUNT` (asset
      movement, not P&L)
    * repayment → :data:`REPAYMENT_CLEARING_ACCOUNT` (non-income equity clearing)
    * every other slug → ``Expenses:<Title>`` — a credit on an expense slug is
      a **reversal** that nets against the expense root, never relabelled as
      income. This is the key difference from the direction-only rooting.

    Operator ``category_mappings`` overrides win for *any* slug (including the
    special ones) and are validated as backend account names. ``unknown``/blank
    maps to ``Expenses:Unknown`` (a suspense contra) and is counted in the
    report's ``unknown_count``.
    """
    slug = (category or "").strip().lower() or "unknown"
    override = config.category_mappings.get(slug)
    if override is not None:
        return _finalize_name(
            override, is_default=False, backend=backend, label=f"category {slug!r}"
        )
    kind = _category_kind(slug)
    title = _title_segment(slug)
    if kind == KIND_INVESTMENT:
        raw = INVESTMENT_UNALLOCATED_ACCOUNT
    elif kind == KIND_REPAYMENT:
        raw = REPAYMENT_CLEARING_ACCOUNT
    elif kind == KIND_INCOME:
        raw = f"Income:{title}"
    elif kind == KIND_CONTRA_EXPENSE:
        raw = f"Expenses:{title}"
    elif kind == KIND_UNKNOWN:
        raw = "Expenses:Unknown"
    else:
        raw = f"Expenses:{title}"
    return _finalize_name(
        raw, is_default=True, backend=backend, label=f"category {slug!r}"
    )


def _card_clearing_account(config: PaisaProjectionConfig, backend: str) -> str:
    """The liability a card payment posts against, resolved for the backend.

    Defaults to :data:`CARD_PAYMENT_CLEARING` (normalized for beancount); an
    operator override is validated strictly.
    """
    override = config.category_mappings.get(CREDIT_CARD_PAYMENT_SLUG)
    raw = override if override is not None else CARD_PAYMENT_CLEARING
    return _finalize_name(
        raw,
        is_default=override is None,
        backend=backend,
        label="credit_card_payment",
    )


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

    Opening balances are always INR: a reliable original commodity is not
    available from a snapshot or running balance, so none is fabricated.

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
    kind = kind_override or _category_kind(slug)
    if contra_override is not None:
        contra = contra_override
    else:
        contra = _contra_account(txn.category, txn.direction, config, backend)
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
    resolved_card_account: LedgerAccount | None = None,
) -> ProjectedEntry:
    """A card-payment bank debit: bank asset ↓, card liability ↓, in ``commodity``.

    **Card resolution** (requirement: never fuzzy match):

    * If ``resolved_card_account`` is provided (the bank row has an explicit
      ``card_id`` or an exact card mask that maps to a selected card account),
      the payment posts to that specific liability.
    * Otherwise it posts to the generic clearing liability
      (:data:`CARD_PAYMENT_CLEARING`) and the entry carries
      ``dashboard_card_resolution=unresolved`` metadata so the operator knows
      the pairing was not determined.

    Only invoked for an ASSET (bank) account: the caller skips card-side
    ``credit_card_payment`` legs so they are not misposted here.
    """
    amount = Decimal(txn.amount)
    date = txn.transaction_date
    assert date is not None  # caller filters transaction_date.is_not(None)
    if resolved_card_account is not None:
        liability_name = resolved_card_account.name
        card_ids = (
            txn.card_id,
            resolved_card_account.account_id,
        )
        extra = (("dashboard_card_resolution", "resolved"),)
    else:
        liability_name = _card_clearing_account(config, backend)
        card_ids = (txn.card_id,) if txn.card_id is not None else ()
        extra = (("dashboard_card_resolution", "unresolved"),)
    meta = _txn_meta(
        txn,
        KIND_CARD_PAYMENT,
        account_ids=(bank_account.account_id,),
        card_ids=card_ids,
        extra=extra,
    )
    return ProjectedEntry(
        date=date,
        payee=txn.counterparty or "Card Payment",
        txn_ids=(txn.id,) if txn.id is not None else (),
        postings=(
            LedgerPosting(account=liability_name, amount=amount, commodity=commodity),
            LedgerPosting(
                account=bank_account.name, amount=-amount, commodity=commodity
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
# Investment lots
# ---------------------------------------------------------------------------

#: ``paisa.project_investments`` gates the entire investment-lot projection.
#: Default off; the setting registration lives in the extension manifest (owned
#: elsewhere), so the config reads it with a ``False`` fallback for a DB that
#: has not registered it yet.

# Investment-lot projection is deliberately separate from the bank/cash flow:
# a lot posts ONLY to ``Assets:Investments:<instrument>`` against a dedicated
# ``Equity:Opening Balances:Investment`` contra, and NO bank/cash leg is
# inferred. This keeps lots from double-counting a bank balance.
#
# Caveat an operator should know: the bank account *purchase* that funded an
# MF acquisition is still a dashboard Transaction. Unless its category is
# mapped (via ``paisa.category_mappings``) to the investment/equity account,
# the bank projection emits it as an ordinary expense while the lot projection
# also records the holding — the same rupee appears twice. Investment
# projection is OFF by default precisely so this is opt-in, and the report's
# diagnostics let an operator see what was projected before relying on it.


async def _load_investment_lot_entries(
    session: AsyncSession,
) -> tuple[tuple[InvestmentLotEntry, ...], tuple[PriceDirective, ...], set[str]]:
    """Read complete :class:`InvestmentLot` rows and build projection entries.

    Each lot becomes one :class:`InvestmentLotEntry` plus a price directive
    dated at its acquisition with its explicit per-unit cost (the CAS nav).
    The entries are sorted by ``(acquired_on, instrument)`` for a stable render.
    Returns ``(entries, prices, suppressed)`` — empty entries/prices and an
    empty set when there are no lots.

    **Disposal safety.** Instruments whose preserved CAS facts contain a
    disposal/redemption that cannot be truthfully allocated to lots are
    *suppressed* (returned in the ``suppressed`` set): their gross acquisition
    lots are not projected, so the default projection never overstates holdings.
    CAS does not tie a redemption to the acquisition lots it settled, so any
    instrument with a free-standing redemption is suppressed conservatively.

    Read-only: this SELECTs lots the CAS ingest already normalized; it never
    writes a core row, and it never infers a cost/acquisition date a lot does
    not explicitly carry.
    """
    from financial_dashboard.services.investments import (
        get_unresolved_disposal_instruments,
    )

    rows = (
        (
            await session.execute(
                select(InvestmentLot).order_by(
                    InvestmentLot.acquired_on, InvestmentLot.instrument_id
                )
            )
        )
        .scalars()
        .all()
    )
    suppressed = await get_unresolved_disposal_instruments(session)
    entries: list[InvestmentLotEntry] = []
    prices: list[PriceDirective] = []
    for lot in rows:
        if lot.instrument_id in suppressed:
            # Conservative: do not project gross acquisitions whose holdings
            # may already have been partially redeemed.
            continue
        # Build non-sensitive CAS provenance metadata for the lot entry.
        lot_meta: list[tuple[str, str]] = [
            ("dashboard_kind", KIND_LOT),
            (
                "dashboard_instrument",
                sanitize_meta_value(lot.instrument_id) or "unknown",
            ),
            ("dashboard_acquired_on", str(lot.acquired_on)),
        ]
        if lot.cas_upload_id is not None:
            lot_meta.append(("dashboard_cas_upload_id", str(lot.cas_upload_id)))
        if lot.source_ref:
            lot_meta.append(
                ("dashboard_source_ref", sanitize_meta_value(lot.source_ref))
            )
        if lot.reference:
            lot_meta.append(("dashboard_reference", sanitize_meta_value(lot.reference)))
        entries.append(
            InvestmentLotEntry(
                instrument=sanitize_commodity(lot.instrument_id),
                instrument_name=lot.instrument_name,
                quantity=Decimal(lot.quantity),
                unit_cost=Decimal(lot.unit_cost),
                cost_basis=Decimal(lot.cost_basis),
                currency=lot.currency,
                acquired_on=lot.acquired_on,
                cas_upload_id=lot.cas_upload_id,
                source_ref=lot.source_ref,
                reference=lot.reference,
                meta=tuple(lot_meta),
            )
        )
        # The explicit per-unit price (CAS nav) as of the acquisition date — a
        # truthful "as-of" price fact, not a synthesized market quote.
        prices.append(
            PriceDirective(
                date=lot.acquired_on,
                currency=sanitize_commodity(lot.instrument_id),
                rate=Decimal(lot.unit_cost),
                unit=lot.currency,
            )
        )
    return tuple(entries), tuple(prices), suppressed


async def _investment_excluded_reasons(session: AsyncSession) -> tuple[str, ...]:
    """Deduplicated stable reason labels for CAS facts excluded from lots.

    Delegates to the investment service, which recomputes exclusions from the
    preserved raw payloads — so the diagnostic reflects the current lot
    classification without a separate persisted store.
    """
    from financial_dashboard.services.investments import get_incomplete_reasons

    exclusions = await get_incomplete_reasons(session)
    seen: dict[str, None] = {}
    for excl in exclusions:
        seen.setdefault(excl.reason, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Card payment resolution (exact card_id / exact mask only — never fuzzy)
# ---------------------------------------------------------------------------


async def _load_card_resolution_maps(
    session: AsyncSession,
    accounts_by_id: dict[int, LedgerAccount],
) -> tuple[dict[int, LedgerAccount], dict[str, LedgerAccount]]:
    """Build ``card_id → liability account`` and ``card_mask → liability`` maps.

    Only card-type (liability) accounts that are in the selected set are
    considered. A bank-side card payment resolves to a specific liability ONLY
    when the row carries an explicit ``card_id`` whose Card belongs to a
    selected account, OR an exact ``card_mask`` match. **Never** fuzzy: a
    partial or near-miss mask does not resolve.
    """
    liability_accounts = {
        aid: acct for aid, acct in accounts_by_id.items() if acct.kind == "liability"
    }
    if not liability_accounts:
        return ({}, {})
    card_rows = (
        (
            await session.execute(
                select(Card).where(Card.account_id.in_(set(liability_accounts.keys())))
            )
        )
        .scalars()
        .all()
    )
    by_card_id: dict[int, LedgerAccount] = {}
    by_mask: dict[str, LedgerAccount] = {}
    for card in card_rows:
        acct = liability_accounts.get(card.account_id)
        if acct is None:
            continue
        if card.id is not None:
            by_card_id[card.id] = acct
        if card.card_mask:
            by_mask[card.card_mask] = acct
    return (by_card_id, by_mask)


def _resolve_card_for_payment(
    txn: Transaction,
    by_card_id: dict[int, LedgerAccount],
    by_mask: dict[str, LedgerAccount],
) -> LedgerAccount | None:
    """Resolve a card-payment bank leg to a specific selected liability.

    Exact-match only: ``txn.card_id`` → ``by_card_id``, or ``txn.card_mask``
    → ``by_mask``. Returns ``None`` (generic clearing) when no exact match —
    never a fuzzy/text-similarity guess.
    """
    if txn.card_id is not None:
        acct = by_card_id.get(txn.card_id)
        if acct is not None:
            return acct
    mask = (txn.card_mask or "").strip()
    if mask:
        acct = by_mask.get(mask)
        if acct is not None:
            return acct
    return None


# ---------------------------------------------------------------------------
# Investment funding double-count prevention (conservative, no fuzzy matching)
# ---------------------------------------------------------------------------


class _InvestmentFundingMap(NamedTuple):
    """Lookup structures for provable lot↔bank-investment funding links.

    * ``by_ref``: lot reference/source_ref → list of (instrument, cost_basis).
    * ``by_date_amount``: (acquired_on, cost_basis) → list of instruments.
    * ``instruments``: the set of all emitted-lot instruments.

    A bank investment transaction provably funds a lot when:
      (a) its ``reference_number`` exactly matches a lot's ``reference`` or
          ``source_ref``; OR
      (b) its ``(transaction_date, amount)`` exactly matches a lot's
          ``(acquired_on, cost_basis)`` AND that key is deterministic (exactly
          one lot instrument matches).

    When neither holds but a potential date-or-amount collision exists, the lot
    is suppressed conservatively (the funding is not provably deterministic, so
    emitting both would risk double-counting the investment asset).
    """

    by_ref: dict[str, list[tuple[str, Decimal]]]
    by_date_amount: dict[tuple[datetime.date, Decimal], list[str]]
    instruments: set[str]


def _build_funding_map(
    lot_entries: tuple[InvestmentLotEntry, ...],
) -> _InvestmentFundingMap:
    """Build the provable-funding lookup from emitted lots."""
    by_ref: dict[str, list[tuple[str, Decimal]]] = {}
    by_date_amount: dict[tuple[datetime.date, Decimal], list[str]] = {}
    instruments: set[str] = set()
    for lot in lot_entries:
        instruments.add(lot.instrument)
        key = (lot.acquired_on, Decimal(lot.cost_basis).quantize(Decimal("0.01")))
        by_date_amount.setdefault(key, []).append(lot.instrument)
        for ref in (lot.reference, lot.source_ref):
            if ref:
                by_ref.setdefault(ref, []).append(
                    (lot.instrument, Decimal(lot.cost_basis))
                )
    return _InvestmentFundingMap(by_ref, by_date_amount, instruments)


def _check_investment_funding(
    txn: Transaction, fmap: _InvestmentFundingMap
) -> tuple[str | None, Decimal | None]:
    """Return ``(instrument, amount)`` if ``txn`` provably funds an emitted lot.

    Returns ``(None, None)`` when not provable. The rules (no fuzzy matching):

    1. Exact reference: ``txn.reference_number`` matches a lot's ``reference``
       or ``source_ref``. If the ref maps to exactly one instrument, it's
       provable. **Do not** early-return when the ref maps to several
       instruments — fall through to the deterministic date+amount check below,
       which may still disambiguate. Early-returning on a shared reference would
       skip that disambiguation and leave the bank leg in a double-count window
       (emitted as an ordinary investment whose lot is also projected).
    2. Exact date+amount: ``(txn.transaction_date, txn.amount)`` matches a
       lot's ``(acquired_on, cost_basis)`` AND exactly one instrument shares
       that key (deterministic).

    The amount is returned so the caller can verify the full funding match.
    """
    ref = (txn.reference_number or "").strip()
    if ref and ref in fmap.by_ref:
        matches = fmap.by_ref[ref]
        if len(matches) == 1:
            instr, amt = matches[0]
            return (instr, amt)
        # Multiple instruments share this ref — ambiguous on the reference
        # alone. Fall through: a deterministic exact date+amount match may
        # still single out one instrument and remap the bank leg provably.
    date = txn.transaction_date
    if date is not None:
        key = (date, Decimal(txn.amount).quantize(Decimal("0.01")))
        matches = fmap.by_date_amount.get(key)
        if matches and len(matches) == 1:
            return (matches[0], Decimal(txn.amount).quantize(Decimal("0.01")))
    return (None, None)


def _ambiguous_funding_instruments(
    txn: Transaction, fmap: _InvestmentFundingMap
) -> set[str]:
    """Instruments with a *potential* (but not provable) funding link to ``txn``.

    Used to suppress lots conservatively when a bank investment transaction
    shares a reference, date, or amount with a lot but the link is not
    provably deterministic (so emitting both would risk double-counting the
    investment asset). Reached only when :func:`_check_investment_funding`
    returned ``(None, None)`` — i.e. no single-instrument reference and no
    deterministic exact date+amount match — so every branch here is the
    conservative fallback for a *potential* link.
    """
    potential: set[str] = set()
    # A reference shared by multiple instruments is a potential-but-not-provable
    # funding link: the bank leg may fund any of them, and we cannot tell which.
    # Suppress every instrument sharing the ref so none is double-counted.
    ref = (txn.reference_number or "").strip()
    if ref and ref in fmap.by_ref and len(fmap.by_ref[ref]) != 1:
        for instr, _amt in fmap.by_ref[ref]:
            potential.add(instr)
    date = txn.transaction_date
    if date is not None:
        amount = Decimal(txn.amount).quantize(Decimal("0.01"))
        # Same date OR same amount (but not the deterministic exact pair, which
        # _check_investment_funding would have caught as provable).
        for (lot_date, lot_amt), instrs in fmap.by_date_amount.items():
            if lot_date == date or lot_amt == amount:
                if not (lot_date == date and lot_amt == amount and len(instrs) == 1):
                    potential.update(instrs)
    return potential


def _is_lot_price(price: PriceDirective, lot_instruments: set[str]) -> bool:
    """True if a price directive was contributed by a lot (vs. an FX rate).

    Lot price directives carry an instrument commodity (ISIN) rather than an
    ISO currency code. The caller passes the **full** set of emitted-lot
    instruments (``fmap.instruments``, before any funding suppression) so a
    suppressed lot's price is identified and dropped — passing the *filtered*
    post-suppression set would let a suppressed instrument's price slip through
    (its commodity is absent from the filtered set, so the check would wrongly
    return False and the orphan price directive would survive).
    """
    return price.currency in lot_instruments


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
    if not config.selected_account_ids:
        return _empty_report(config.cutover_date, config.ledger_cli)

    cutover = config.cutover_date
    selected = set(config.selected_account_ids)
    backend = config.ledger_cli

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
        accounts_by_id[acct.id] = _resolve_account(
            acct, config.account_mappings, backend
        )

    # ---- card resolution maps (exact card_id / mask → liability) ------------
    card_by_id, card_by_mask = await _load_card_resolution_maps(session, accounts_by_id)

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
    imprecise = 0
    self_pairs = 0
    investment_funding_remapped = 0
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

    # ---- investment lots (loaded BEFORE the linear pass so funding dedup
    #      can remap bank investment legs that provably fund an emitted lot).
    lot_entries: tuple[InvestmentLotEntry, ...] = ()
    investment_excluded: tuple[str, ...] = ()
    disposal_suppressed: set[str] = set()
    if config.project_investments:
        (
            lot_entries,
            lot_prices,
            disposal_suppressed,
        ) = await _load_investment_lot_entries(session)
        prices.extend(lot_prices)
        investment_excluded = await _investment_excluded_reasons(session)
        if disposal_suppressed:
            investment_excluded = tuple(
                dict.fromkeys((*investment_excluded, "disposal_history_unresolved"))
            )

    fmap = _build_funding_map(lot_entries)
    funding_suppressed: set[str] = set()

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
            resolved_card = _resolve_card_for_payment(txn, card_by_id, card_by_mask)
            entries.append(
                _build_card_payment_entry(
                    txn,
                    account,
                    config,
                    backend,
                    commodity,
                    resolved_card_account=resolved_card,
                )
            )
            card_payments += 1
            if resolved_card is not None:
                card_payments_resolved += 1
            else:
                card_payments_unresolved += 1
        else:
            if category in ("", "unknown"):
                unknown += 1
            if category in IMPRECISE_CATEGORY_SLUGS:
                imprecise += 1
            # Investment funding double-count prevention: if this investment-
            # category transaction provably funds an emitted lot, remap the
            # bank leg's contra to INVESTMENT_EQUITY_OPENING so the investment
            # asset is counted once (in the lot), not twice. If there is a
            # potential but not provable link, suppress the lot conservatively.
            contra_override = None
            kind_override = None
            if category in INVESTMENT_CATEGORY_SLUGS and lot_entries:
                instr, _funding_amt = _check_investment_funding(txn, fmap)
                if instr is not None:
                    contra_override = _finalize_name(
                        INVESTMENT_EQUITY_OPENING,
                        is_default=True,
                        backend=backend,
                        label="investment_funding_remap",
                    )
                    kind_override = KIND_INVESTMENT
                    investment_funding_remapped += 1
                else:
                    # Potential but not provable: suppress the matching lot(s)
                    # so we never emit both a lot AND an unresolved bank leg
                    # pointing at Assets:Investments.
                    funding_suppressed.update(_ambiguous_funding_instruments(txn, fmap))
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

    # Apply investment-funding suppression: lots whose funding link is potential
    # but not provably deterministic are removed so the investment asset is
    # never double-counted (the bank leg still captures the bank decrease).
    if funding_suppressed:
        lot_entries = tuple(
            lot for lot in lot_entries if lot.instrument not in funding_suppressed
        )
        # Drop every lot price directive (identified against the FULL instrument
        # set, not the filtered one) and re-derive prices from the surviving
        # lots — so a suppressed lot's price directive cannot linger as an
        # orphan. FX price directives (ISO currency codes) are never lot prices
        # and are left untouched.
        prices = [p for p in prices if not _is_lot_price(p, fmap.instruments)] + [
            PriceDirective(
                date=lot.acquired_on,
                currency=sanitize_commodity(lot.instrument),
                rate=Decimal(lot.unit_cost),
                unit=lot.currency,
            )
            for lot in lot_entries
        ]
        investment_excluded = tuple(
            dict.fromkeys((*investment_excluded, "investment_funding_unresolved"))
        )

    # Deduplicate price directives by (date, currency): the rate is a pure
    # function of (currency, date) from the configured map, so this collapses
    # same-day same-currency rows into one directive per backend file.
    deduped_prices = _dedupe_prices(prices)
    doc = LedgerDocument(
        cutover_date=cutover,
        openings=tuple(openings_list),
        entries=tuple(entries),
        accounts_declared=tuple(declared),
        price_directives=deduped_prices,
        lot_postings=lot_entries,
    )
    journal = render_document_for_backend(doc, backend)

    foreign_count = sum(1 for e in entries if e.currency != INR)
    kind_counts: dict[str, int] = {}
    for e in entries:
        kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1
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
        investment_lot_count=len(lot_entries),
        investment_excluded=investment_excluded,
        investment_disposal_unresolved=tuple(sorted(disposal_suppressed)),
        imprecise_count=imprecise,
        card_payments_resolved=card_payments_resolved,
        card_payments_unresolved=card_payments_unresolved,
        investment_funding_remapped=investment_funding_remapped,
        investment_funding_unresolved=tuple(sorted(funding_suppressed)),
        kind_counts=kind_counts,
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
