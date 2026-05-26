"""Transaction-to-account linker.

Resolves account_id and card_id on Transaction rows by matching the
card_mask / account_mask emitted by the parser against the accounts and
cards tables.

Indian bank emails and SMSes use at least these mask formats:

    "XX1234"               -- short X-prefix, last-4 suffix
    "xx1234"               -- same, lowercase
    "XX123"                -- short X-prefix, last-3 (ICICI savings SMSes)
    "XXXXXXX1234"          -- long X-prefix, last-4 suffix
    "0000 XXXX XXXX 1234"  -- full 16-digit card layout with spaces
    "12XXXXXX1234"         -- partial mask, digits at both ends
    "1234"                 -- bare last-4 (SBI, some bare-numeric forms)

The matcher works on the **digit run** of the incoming mask: it strips
non-digit characters, then suffix-matches against the stored digits of
each account/card scoped to the same bank. The stored digit string
can be longer than the incoming one — whichever is shorter must be a
suffix of the longer. So an account_number `000000001234` matches
incoming mask `XX234` because `234` is a suffix of `000000001234`.
A minimum of 3 trailing digits is required to avoid pathological
matches.

Lookup is scoped by bank, so a card in one bank cannot collide with
an unrelated account in another bank that happens to share the same
trailing digits.

Lookup precedence (per transaction):
  1. card_mask  -> cards table  (sets both card_id AND account_id)
  2. card_mask  -> accounts table  (debit cards stored as account_number)
  3. account_mask -> accounts table
  4. bank-only  -> accounts table  (only when no mask at all and exactly one
                                     account exists for that bank)

When more than one account/card within the same bank shares the matched
suffix, the linker refuses to guess and leaves the row unlinked.

Batch usage:

    ctx = await build_link_context(session)
    for txn in orphan_transactions:
        link_transaction(ctx, txn)
    await session.commit()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, Card, Transaction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


MIN_MASK_DIGITS = 3
"""Minimum trailing-digit count required to attempt a match.

ICICI savings SMSes carry as few as 3 digits in the account_mask
(e.g. ``XX234``). Shorter masks produce dangerous false positives
(e.g. matching every account ending in ``1``).
"""


def _trailing_digits(mask: str) -> str:
    """Return every digit in *mask*, in order, with non-digits stripped.

    For an incoming mask the matcher uses the full digit string and asks
    "is this a suffix of the stored account number?". For an existing
    account, we store the same digit string and let the suffix-match
    happen on the *stored* side.

    Examples
    --------
    >>> _trailing_digits("XX1234")        == "1234"
    >>> _trailing_digits("xx5678")        == "5678"
    >>> _trailing_digits("XX234")         == "234"
    >>> _trailing_digits("0000 XXXX XXXX 1234") == "00001234"
    >>> _trailing_digits("1234")          == "1234"
    >>> _trailing_digits("")              == ""
    """
    return re.sub(r"[^0-9]", "", mask)


# ---------------------------------------------------------------------------
# Link context (preloaded lookup tables)
# ---------------------------------------------------------------------------


@dataclass
class LinkContext:
    """Preloaded lookup structures built once and reused for a batch.

    cards_by_bank:
        bank (lowercase) -> list of (digit_string, account_id, card_id)
        One entry per Card. digit_string is the trailing digit run of
        Card.card_mask. Matched by suffix-equality against the incoming
        transaction's mask digits.

    accounts_by_bank_with_digits:
        bank (lowercase) -> list of (digit_string, account_id)
        One entry per Account that has a non-empty account_number.
        Matched the same way as cards.

    accounts_by_bank:
        bank (lowercase) -> list[account_id]
        Used only for the maskless bank-only fallback: if a transaction
        carries no mask at all and exactly one account exists for that
        bank, we link to it.
    """

    cards_by_bank: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    accounts_by_bank_with_digits: dict[str, list[tuple[str, int]]] = field(
        default_factory=dict
    )
    accounts_by_bank: dict[str, list[int]] = field(default_factory=dict)
    account_types: dict[int, str] = field(default_factory=dict)
    """account_id -> account.type (e.g. 'bank_account', 'credit_card'). Used by
    the bank-only fallback to disambiguate between e.g. Slice Savings vs
    Slice CC when the SMS body carries no mask."""


def _expected_account_type(email_type: str | None) -> str | None:
    """Infer the Account.type a transaction's email_type implies.

    Returns:
      - "credit_card" for any '*_cc_*' email_type
      - "bank_account" for any '*_account_*' or '*_dc_*' email_type (the
        debit-card-on-savings shape)
      - None when the email_type doesn't carry an account-vs-card hint
    """
    if not email_type:
        return None
    if "_cc_" in email_type:
        return "credit_card"
    if "_account_" in email_type or "_dc_" in email_type:
        return "bank_account"
    return None


def _suffix_match(incoming: str, stored: str) -> bool:
    """True iff the shorter of *incoming* and *stored* is a suffix of
    the longer.

    Both are digit-only strings (non-digit chars already stripped).
    """
    if not incoming or not stored:
        return False
    if len(incoming) <= len(stored):
        return stored.endswith(incoming)
    return incoming.endswith(stored)


async def build_link_context(session: AsyncSession) -> LinkContext:
    """Load all accounts and cards from the DB and build lookup tables.

    Call this once before processing a batch of transactions. The
    returned LinkContext is a plain Python object — no further DB
    queries are needed until you want to refresh it.
    """
    ctx = LinkContext()

    accounts = (await session.execute(select(Account))).scalars().all()
    for acct in accounts:
        bank_key = acct.bank.strip().lower()
        ctx.accounts_by_bank.setdefault(bank_key, []).append(acct.id)
        if acct.type:
            ctx.account_types[acct.id] = acct.type

        if acct.account_number:
            digits = _trailing_digits(acct.account_number)
            if digits:
                ctx.accounts_by_bank_with_digits.setdefault(bank_key, []).append(
                    (digits, acct.id)
                )

    cards = (await session.execute(select(Card))).scalars().all()
    for card in cards:
        digits = _trailing_digits(card.card_mask)
        if not digits:
            continue
        # Look up the card's owning account to get its bank.
        acct = next((a for a in accounts if a.id == card.account_id), None)
        if acct is None:
            continue
        bank_key = acct.bank.strip().lower()
        ctx.cards_by_bank.setdefault(bank_key, []).append(
            (digits, card.account_id, card.id)
        )

    logger.debug(
        "LinkContext built: %d banks with cards, %d banks with accounts, %d banks total",
        len(ctx.cards_by_bank),
        len(ctx.accounts_by_bank_with_digits),
        len(ctx.accounts_by_bank),
    )
    return ctx


def _find_card_match(
    ctx: LinkContext, bank_key: str, incoming_digits: str
) -> tuple[int, int] | None:
    """Find a card in *bank_key* whose stored digits suffix-match
    *incoming_digits*.

    Returns (account_id, card_id) on a unique match, None otherwise.
    Logs a warning when multiple candidates suffix-match (ambiguous).
    """
    matches = [
        (acct_id, card_id)
        for stored, acct_id, card_id in ctx.cards_by_bank.get(bank_key, [])
        if _suffix_match(incoming_digits, stored)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Ambiguous card-mask match in bank %r: incoming digits %r matched "
            "%d cards %r — refusing to link.",
            bank_key,
            incoming_digits,
            len(matches),
            matches,
        )
    return None


def _find_account_match(
    ctx: LinkContext, bank_key: str, incoming_digits: str
) -> int | None:
    """Find an account in *bank_key* whose account_number digits
    suffix-match *incoming_digits*.

    Returns account_id on a unique match, None otherwise.
    Logs a warning on ambiguity.
    """
    matches = [
        acct_id
        for stored, acct_id in ctx.accounts_by_bank_with_digits.get(bank_key, [])
        if _suffix_match(incoming_digits, stored)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Ambiguous account-mask match in bank %r: incoming digits %r matched "
            "%d accounts %r — refusing to link.",
            bank_key,
            incoming_digits,
            len(matches),
            matches,
        )
    return None


# ---------------------------------------------------------------------------
# Single-transaction linker
# ---------------------------------------------------------------------------


def link_transaction(ctx: LinkContext, txn: Transaction) -> bool:
    """Attempt to set account_id (and card_id) on *txn* using *ctx*.

    Mutates *txn* in place.  The caller is responsible for committing the
    session.

    Returns True if a link was established, False otherwise.

    Precedence
    ----------
    1. card_mask -> cards table
       Best match: identifies the exact physical card, and the card row
       carries a FK to its parent account, so both card_id and account_id
       are set.  This is the only path that populates card_id.

    2. card_mask -> accounts table
       Fallback for cards that are stored directly as Account rows (e.g.
       a debit card whose account_number IS the last-4).  Sets account_id
       only.

    3. account_mask -> accounts table
       Savings / current account masks like "xx5678".

    4. bank-only
       When neither mask is present and the bank has exactly one account
       registered, we link to it.  This covers banks that never include a
       mask in their emails (some UPI alert formats).
    """
    if txn.account_id is not None:
        # Already linked -- nothing to do.
        return True

    bank_key = txn.bank.strip().lower() if txn.bank else ""

    # ---- 1. card_mask -> cards table ----
    if txn.card_mask:
        digits = _trailing_digits(txn.card_mask)
        if digits and len(digits) >= MIN_MASK_DIGITS:
            hit = _find_card_match(ctx, bank_key, digits)
            if hit is not None:
                acct_id, card_id = hit
                txn.account_id = acct_id
                txn.card_id = card_id
                logger.debug(
                    "txn %s: linked via cards table (mask=%r -> digits=%s, account=%s card=%s)",
                    txn.id,
                    txn.card_mask,
                    digits,
                    acct_id,
                    card_id,
                )
                return True

    # ---- 2. card_mask -> accounts table ----
    if txn.card_mask:
        digits = _trailing_digits(txn.card_mask)
        if digits and len(digits) >= MIN_MASK_DIGITS:
            hit = _find_account_match(ctx, bank_key, digits)
            if hit is not None:
                txn.account_id = hit
                logger.debug(
                    "txn %s: linked via accounts table by card_mask (mask=%r -> digits=%s, account=%s)",
                    txn.id,
                    txn.card_mask,
                    digits,
                    hit,
                )
                return True

    # ---- 3. account_mask -> accounts table ----
    if txn.account_mask:
        digits = _trailing_digits(txn.account_mask)
        if digits and len(digits) >= MIN_MASK_DIGITS:
            hit = _find_account_match(ctx, bank_key, digits)
            if hit is not None:
                txn.account_id = hit
                logger.debug(
                    "txn %s: linked via accounts table by account_mask (mask=%r -> digits=%s, account=%s)",
                    txn.id,
                    txn.account_mask,
                    digits,
                    hit,
                )
                return True

    # ---- 4. bank-only fallback ----
    # Used when the message body carries no mask (typical for CC bill-paid
    # / statement-ready alerts, and some maskless payment-received shapes).
    # The candidate set is narrowed by the account *type* the email_type
    # implies — '*_cc_*' restricts to credit_card accounts, '*_account_*'
    # / '*_dc_*' to bank_account. This disambiguates a bank that has both
    # a savings account and a CC registered without needing a mask.
    if not txn.card_mask and not txn.account_mask:
        bank_key = txn.bank.strip().lower()
        acct_ids = ctx.accounts_by_bank.get(bank_key, [])
        expected_type = _expected_account_type(txn.email_type)
        if expected_type is not None:
            acct_ids = [
                a for a in acct_ids if ctx.account_types.get(a) == expected_type
            ]
        if len(acct_ids) == 1:
            txn.account_id = acct_ids[0]
            logger.debug(
                "txn %s: linked via bank-only fallback (bank=%r, type=%r, account=%s)",
                txn.id,
                txn.bank,
                expected_type,
                txn.account_id,
            )
            return True
        if len(acct_ids) > 1:
            logger.warning(
                "txn %s: bank-only fallback skipped -- %d candidates for "
                "bank %r type=%r (no card_mask / account_mask; leaving unlinked "
                "to avoid wrong attribution)",
                txn.id,
                len(acct_ids),
                txn.bank,
                expected_type,
            )

    logger.debug(
        "txn %s: no link found (bank=%r card_mask=%r account_mask=%r)",
        txn.id,
        txn.bank,
        txn.card_mask,
        txn.account_mask,
    )
    return False


# ---------------------------------------------------------------------------
# Convenience: relink all orphans in one shot
# ---------------------------------------------------------------------------


async def relink_orphans(session: AsyncSession) -> tuple[int, int]:
    """Link every unlinked transaction in the DB.

    Returns (linked_count, remaining_count).

    Useful for seed_accounts.py and one-off repair scripts.
    """
    ctx = await build_link_context(session)

    orphans = (
        (
            await session.execute(
                select(Transaction).where(Transaction.account_id.is_(None))
            )
        )
        .scalars()
        .all()
    )

    linked = sum(1 for txn in orphans if link_transaction(ctx, txn))
    await session.commit()

    remaining = (
        (
            await session.execute(
                select(Transaction).where(Transaction.account_id.is_(None))
            )
        )
        .scalars()
        .all()
    )

    logger.info(
        "relink_orphans: linked %d, %d still unlinked",
        linked,
        len(remaining),
    )
    return linked, len(remaining)
