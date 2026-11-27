"""Payment view-model for the CC statement detail page.

The statement page shows three things for one CC cycle:
- a payment-status line (paid / total / remaining),
- the real credit transactions received against the cycle (settled), and
- provisional bill-payment SMS not yet in the ledger (pending), each with a
  manual "settle" action.

A provisional CC-payment SMS is notify-only in the pipeline: it makes no
Transaction row. So the pending set is not a stored table. It is derived live
by re-parsing the account's unlinked SMS in the cycle window and keeping the
ones the parser calls a provisional credit. Once a row settles it links to a
Transaction and drops off the pending set on its own. This is deliberately
self-healing: no staging table to clean up.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bank_sms_parser import ParseError, UnsupportedSmsTypeError, parse_sms

from financial_dashboard.db import (
    Account,
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.cc_cycle import (
    CycleWindow,
    cc_cycle_window,
    transactions_in_cycle,
)
from financial_dashboard.services.cc_disambiguation import (
    is_cc_payment_received_email,
)
from financial_dashboard.services.sms_pipeline import parsed_sms_to_txn_data
from financial_dashboard.services.statements.cc import parse_cc_amount

logger = logging.getLogger(__name__)

# A payment on a later statement can post a few days after its provisional SMS.
SETTLE_DAYS = 3
# A statement credit that is a bill payment, by its narration. Refunds and
# cashback are credits too, so a narration check keeps them from hiding a
# provisional that has the same amount. A narration can run words together,
# as in "PAYMENTNet Banking", so only the start of a word is checked.
PAYMENT_NARRATION = re.compile(
    r"\b(PAYMENT|PAYZAPP|BPPY|NETBANKING|TELE TRANSFER)", re.IGNORECASE
)


@dataclass(frozen=True)
class SettledPayment:
    """A real credit transaction counted against this cycle."""

    txn_id: int
    date: str
    amount: str
    counterparty: str


@dataclass(frozen=True)
class PendingPayment:
    """A provisional bill-payment SMS not yet in the ledger."""

    sms_id: int
    amount: str
    date: str
    card_mask: str
    available_limit: str | None


@dataclass(frozen=True)
class PaymentsView:
    """Everything the statement page's Payments panel needs."""

    settled: list[SettledPayment]
    pending: list[PendingPayment]
    remaining: str | None


def _fmt_amount(value) -> str:
    return f"{Decimal(str(value)):,.2f}"


async def _settled_payments(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
) -> list[SettledPayment]:
    """Real credit transactions that count toward this cycle's paid amount.

    Mirrors ``_qualifying_payment_credit_sum``: credit rows the classifier calls
    a CC bill-payment, in the same due-date-anchored cycle window.
    """
    rows = (
        (
            await session.execute(
                select(Transaction)
                .where(
                    Transaction.account_id == upload.account_id,
                    Transaction.direction == "credit",
                    transactions_in_cycle(window),
                )
                .order_by(Transaction.transaction_date.asc(), Transaction.id.asc())
            )
        )
        .scalars()
        .all()
    )

    settled: list[SettledPayment] = []
    for row in rows:
        if not is_cc_payment_received_email(row.email_type):
            continue
        settled.append(
            SettledPayment(
                txn_id=row.id,
                date=row.transaction_date.isoformat() if row.transaction_date else "",
                amount=_fmt_amount(row.amount),
                counterparty=row.counterparty or "",
            )
        )
    return settled


async def _pending_payments(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
) -> list[PendingPayment]:
    """Provisional bill-payment SMS for this card, not yet in the ledger.

    Fetches the account's unlinked SMS in the cycle window, then re-parses each
    body. Keeps only the ones the parser calls a provisional credit that the
    classifier reads as a CC bill-payment. Re-parsing is required: the parser's
    ledger role is not stored on the SMS row. The candidate set is small (one
    card's unlinked SMS in one cycle), so parsing on page load is cheap.

    A provisional whose payment is already in the ledger must not show as
    pending. That row is a settlement SMS or email, or, when the bank sends no
    settlement message, a payment line of a later statement. It is a separate
    row, so the provisional stays unlinked. Each provisional pairs with one
    such credit of the same amount, dated the same day or up to
    ``SETTLE_DAYS`` later, and each credit pairs once. A provisional dated
    before the window pays the previous bill, so it shows on that bill.
    """
    account = await session.get(Account, upload.account_id)
    if account is None:
        return []

    # ``received_at`` is UTC; the cycle bounds are calendar dates whose real
    # day is IST. A payment near IST-midnight is stored on the adjacent UTC
    # date, so a tight UTC-midnight window would drop it. Pad the window by a
    # day on each edge to cover the ~5.5h skew. Over-inclusion is safe: a
    # provisional pulled in from an adjacent cycle still needs a transaction
    # identity match, a matching card, and ``transaction_id IS NULL``;
    # the worst case is showing a genuinely-unsettled neighbour payment, which
    # settling attributes to the correct cycle anyway.
    lower = datetime.combine(
        window.start - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    conditions = [
        func.lower(SmsMessage.bank) == account.bank.lower(),
        SmsMessage.transaction_id.is_(None),
        SmsMessage.received_at >= lower,
    ]
    if window.end is not None:
        upper = datetime.combine(
            window.end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        conditions.append(SmsMessage.received_at < upper)

    rows = (
        (
            await session.execute(
                select(SmsMessage)
                .where(*conditions)
                .order_by(SmsMessage.received_at.asc())
            )
        )
        .scalars()
        .all()
    )

    provisionals = []
    for sms in rows:
        parsed = _parse_provisional_cc_payment(sms)
        if parsed is None:
            continue
        txn_data, ledger_role = parsed
        # Only true provisionals settle here. A restatement echoes a payment
        # another message already carries, so it must not become a ledger row.
        if ledger_role != "provisional":
            continue
        if not is_cc_payment_received_email(txn_data.get("email_type")):
            continue
        # A provisional SMS for a DIFFERENT card on the same bank must not show
        # here. Confirm the parsed card belongs to this account's card set.
        if not _card_belongs_to_account(txn_data.get("card_mask"), account):
            continue
        txn_date = txn_data.get("transaction_date")
        if txn_date is not None and txn_date < window.start:
            continue
        provisionals.append((sms, txn_data))

    unpaired = await _unpaired_provisionals(session, upload.account_id, provisionals)
    pending: list[PendingPayment] = []
    for sms, txn_data in unpaired:
        txn_date = txn_data.get("transaction_date")
        balance = txn_data.get("balance")
        pending.append(
            PendingPayment(
                sms_id=sms.id,
                amount=_fmt_amount(txn_data["amount"]),
                date=txn_date.isoformat() if txn_date else "",
                card_mask=txn_data.get("card_mask") or "",
                available_limit=_fmt_amount(balance) if balance is not None else None,
            )
        )
    return pending


async def _unpaired_provisionals(
    session: AsyncSession,
    account_id: int,
    provisionals: list[tuple[SmsMessage, dict]],
) -> list[tuple[SmsMessage, dict]]:
    """Drop each provisional that pairs with a payment credit in the ledger.

    Pairs in date order. A provisional takes the earliest unused credit of
    the same amount dated from its own day to ``SETTLE_DAYS`` later. So two
    provisionals of one amount and one credit leave one provisional.

    Args:
        session: The page session.
        account_id: The card account of the statement.
        provisionals: The provisional SMS with their parsed data.

    Returns:
        The provisionals that no credit pairs with, in date order.
    """
    dated = sorted(
        provisionals,
        key=lambda item: (
            item[1].get("transaction_date") or date.max,
            item[0].received_at,
        ),
    )
    dates = [d for _, data in dated if (d := data.get("transaction_date"))]
    if not dates:
        return dated
    credits = list(
        (
            await session.scalars(
                select(Transaction)
                .where(
                    Transaction.account_id == account_id,
                    Transaction.direction == "credit",
                    Transaction.voided.is_(False),
                    Transaction.transaction_date.between(
                        min(dates), max(dates) + timedelta(days=SETTLE_DAYS)
                    ),
                )
                .order_by(Transaction.transaction_date, Transaction.id)
            )
        ).all()
    )
    payments = [row for row in credits if _is_payment_credit(row)]
    unpaired = []
    for sms, data in dated:
        txn_date = data.get("transaction_date")
        match = next(
            (
                row
                for row in payments
                if txn_date is not None
                and row.amount == data["amount"]
                and row.transaction_date is not None
                and txn_date
                <= row.transaction_date
                <= txn_date + timedelta(days=SETTLE_DAYS)
            ),
            None,
        )
        if match is None:
            unpaired.append((sms, data))
        else:
            payments.remove(match)
    return unpaired


def _is_payment_credit(row: Transaction) -> bool:
    """Say whether a credit row is a card bill payment, not a refund.

    An alert row says so by its type. A statement row says so only by its
    narration, because its category can call a refund a card payment.
    """
    if row.email_type == "cc_statement":
        narration = row.raw_description or row.counterparty or ""
        return bool(PAYMENT_NARRATION.search(narration))
    return is_cc_payment_received_email(row.email_type)


def _parse_provisional_cc_payment(sms: SmsMessage):
    """Re-parse one SMS. Return ``(txn_data, ledger_role)`` or ``None``.

    Side-effect free. Swallows parser errors: a body that no longer parses is
    simply not a pending payment.
    """
    received_at = sms.received_at
    if received_at is not None and received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    try:
        parsed = parse_sms(
            sms.bank, sms.body, sender=sms.sender, received_at=received_at
        )
    except ParseError, UnsupportedSmsTypeError:
        return None

    txn_data = parsed_sms_to_txn_data(parsed, sms)
    if txn_data is None or txn_data.get("direction") != "credit":
        return None
    return txn_data, parsed.ledger_role


def _account_masks(account: Account) -> list[str]:
    masks = [account.account_number or ""]
    masks.extend(card.card_mask or "" for card in getattr(account, "cards", []) or [])
    return masks


def _card_belongs_to_account(
    card_mask: str | None, account: Account, *, strict: bool = False
) -> bool:
    """True if a parsed card mask matches this account or one of its cards.

    ``strict=False`` (display path): a missing or unreadable mask is treated as
    belonging. The pipeline resolves such rows to the single-CC account anyway,
    and excluding them would hide a real pending payment.

    ``strict=True`` (settle path): a missing or unreadable mask must NOT match
    an arbitrary account — settling creates a real credit. Such a mask is
    accepted only when the account has exactly one known card mask (a single-CC
    account where the payment unambiguously belongs), and rejected otherwise.
    """
    readable = [
        m4
        for m in _account_masks(account)
        if (m4 := "".join(ch for ch in m if ch.isdigit())[-4:])
    ]
    last4 = "".join(ch for ch in (card_mask or "") if ch.isdigit())[-4:]
    if not last4:
        if not strict:
            return True
        # Strict: accept a maskless payment only for an unambiguous single card.
        return len(set(readable)) == 1
    return last4 in readable


async def is_settleable_provisional(
    session: AsyncSession, sms: SmsMessage, upload: StatementUpload
) -> bool:
    """True if ``sms`` is a provisional CC bill-payment for ``upload``'s account.

    The settle route accepts a raw SMS id from the form, so it must confirm the
    SMS really is a provisional CC-payment credit for this statement's card
    before promoting it. This rejects a restatement, a non-payment SMS, a
    different card, or an arbitrary id. It does NOT re-check the cycle window:
    settling a slightly out-of-window provisional the user can see is harmless
    and the paid-amount recompute scopes the credit to the right cycle anyway.
    """
    if sms.transaction_id is not None:
        return False
    account = await session.get(Account, upload.account_id)
    if account is None or account.type != "credit_card":
        return False
    if sms.bank.casefold() != account.bank.casefold():
        return False
    parsed = _parse_provisional_cc_payment(sms)
    if parsed is None:
        return False
    txn_data, ledger_role = parsed
    if ledger_role != "provisional":
        return False
    if not is_cc_payment_received_email(txn_data.get("email_type")):
        return False
    return _card_belongs_to_account(txn_data.get("card_mask"), account, strict=True)


async def build_payments_view(
    session: AsyncSession, upload: StatementUpload
) -> PaymentsView:
    """Assemble the Payments panel view-model for one CC statement.

    Returns empty settled/pending and no remaining when the statement has no
    parseable total due (the panel then renders only what it can).
    """
    window = await cc_cycle_window(session, upload)
    settled = await _settled_payments(session, upload, window)
    pending = await _pending_payments(session, upload, window)

    remaining: str | None = None
    if upload.total_amount_due is not None:
        try:
            amount_due = parse_cc_amount(upload.total_amount_due)
        except ValueError, InvalidOperation:
            amount_due = None
        if amount_due is not None:
            paid = Decimal(str(upload.payment_paid_amount or 0))
            remaining = _fmt_amount(max(amount_due - paid, Decimal("0")))

    return PaymentsView(settled=settled, pending=pending, remaining=remaining)
