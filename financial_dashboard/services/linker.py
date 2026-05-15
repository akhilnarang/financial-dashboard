"""Transaction-to-account linker.

Resolves account_id and card_id on Transaction rows by matching the
card_mask / account_mask emitted by the parser against the accounts and
cards tables.

Indian bank emails use at least five distinct mask formats:

    "XX2001"               -- short X-prefix, last-4 suffix
    "xx0298"               -- same, lowercase
    "XXXXXXX8669"          -- long X-prefix, last-4 suffix
    "4611 XXXX XXXX 2002"  -- full 16-digit card layout with spaces
    "5524 XXXX XXXX 2001"  -- same
    "15XXXXXX4006"         -- partial mask, digits at both ends
    "0567"                 -- bare last-4 (SBI, some others)

_last4() strips everything that isn't a digit and returns the trailing
four characters.  That one rule handles all formats.

Lookup precedence (per transaction):
  1. card_mask  -> cards table  (sets both card_id AND account_id)
  2. card_mask  -> accounts table  (addon/debit cards stored as account_number)
  3. account_mask -> accounts table
  4. bank-only  -> accounts table  (only when no mask at all and exactly one
                                     account exists for that bank)

Batch usage (fetcher.py, seed_accounts.py):

    ctx = await build_link_context(session)
    for txn in orphan_transactions:
        link_transaction(ctx, txn)
    await session.commit()

Single-transaction usage (fetcher.py inline, right after INSERT):

    ctx = await build_link_context(session)
    link_transaction(ctx, txn_row)
    # session.commit() happens in the caller
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


def _last4(mask: str) -> str:
    """Return the last 4 digit characters of any mask string.

    Strips all non-digit characters (X, x, spaces, hyphens, …) and
    returns the final 4 digits.  Returns an empty string when fewer than
    4 digits are present so callers can safely guard with ``if digits:``.

    Examples
    --------
    >>> _last4("XX2001")        == "2001"
    >>> _last4("xx0298")        == "0298"
    >>> _last4("XXXXXXX8669")   == "8669"
    >>> _last4("4611 XXXX XXXX 2002") == "2002"
    >>> _last4("15XXXXXX4006")  == "4006"
    >>> _last4("0567")          == "0567"
    >>> _last4("10225478669")   == "8669"   # full account number
    >>> _last4("")              == ""
    """
    digits = re.sub(r"[^0-9]", "", mask)
    return digits[-4:] if len(digits) >= 4 else digits


# ---------------------------------------------------------------------------
# Link context (preloaded lookup tables)
# ---------------------------------------------------------------------------


@dataclass
class LinkContext:
    """Preloaded lookup structures built once and reused for a batch.

    card_by_last4:
        last-4 digits -> (account_id, card_id)
        Populated from the cards table.  When a transaction's card_mask
        matches here we set both fields, correctly attributing addon-card
        spend to the parent account.

    account_by_last4:
        last-4 digits -> account_id
        Populated from the accounts table using each account's
        account_number.  Handles savings-account masks like "xx0298"
        (last-4 of "033325229090298" == "0298") and cards stored as
        account rows with a 4-digit account_number.

    accounts_by_bank:
        bank (lowercase) -> list[account_id]
        Used only for the maskless bank-only fallback: if a transaction
        carries no mask at all and exactly one account exists for that
        bank, we link to it.
    """

    card_by_last4: dict[str, tuple[int, int]] = field(default_factory=dict)
    account_by_last4: dict[str, int] = field(default_factory=dict)
    accounts_by_bank: dict[str, list[int]] = field(default_factory=dict)


async def build_link_context(session: AsyncSession) -> LinkContext:
    """Load all accounts and cards from the DB and build lookup tables.

    Call this once before processing a batch of transactions.  The
    returned LinkContext is a plain Python object -- no further DB
    queries are needed until you want to refresh it.
    """
    ctx = LinkContext()

    # ---- accounts ----
    accounts = (await session.execute(select(Account))).scalars().all()
    for acct in accounts:
        bank_key = acct.bank.strip().lower()
        ctx.accounts_by_bank.setdefault(bank_key, []).append(acct.id)

        if acct.account_number:
            digits = _last4(acct.account_number)
            if digits:
                # Later rows intentionally overwrite earlier ones for the
                # same last-4 within a bank -- the cards table is the
                # authoritative source for card matching anyway.
                ctx.account_by_last4[digits] = acct.id

    # ---- cards ----
    cards = (await session.execute(select(Card))).scalars().all()
    for card in cards:
        digits = _last4(card.card_mask)
        if digits:
            ctx.card_by_last4[digits] = (card.account_id, card.id)

    logger.debug(
        "LinkContext built: %d card entries, %d account entries, %d banks",
        len(ctx.card_by_last4),
        len(ctx.account_by_last4),
        len(ctx.accounts_by_bank),
    )
    return ctx


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
       Savings / current account masks like "xx0298".

    4. bank-only
       When neither mask is present and the bank has exactly one account
       registered, we link to it.  This covers banks that never include a
       mask in their emails (some UPI alert formats).
    """
    if txn.account_id is not None:
        # Already linked -- nothing to do.
        return True

    # ---- 1. card_mask -> cards table ----
    if txn.card_mask:
        digits = _last4(txn.card_mask)
        if digits and digits in ctx.card_by_last4:
            acct_id, card_id = ctx.card_by_last4[digits]
            txn.account_id = acct_id
            txn.card_id = card_id
            logger.debug(
                "txn %s: linked via cards table (mask=%r -> last4=%s, account=%s card=%s)",
                txn.id,
                txn.card_mask,
                digits,
                acct_id,
                card_id,
            )
            return True

    # ---- 2. card_mask -> accounts table ----
    if txn.card_mask:
        digits = _last4(txn.card_mask)
        if digits and digits in ctx.account_by_last4:
            txn.account_id = ctx.account_by_last4[digits]
            logger.debug(
                "txn %s: linked via accounts table by card_mask (mask=%r -> last4=%s, account=%s)",
                txn.id,
                txn.card_mask,
                digits,
                txn.account_id,
            )
            return True

    # ---- 3. account_mask -> accounts table ----
    if txn.account_mask:
        digits = _last4(txn.account_mask)
        if digits and digits in ctx.account_by_last4:
            txn.account_id = ctx.account_by_last4[digits]
            logger.debug(
                "txn %s: linked via accounts table by account_mask (mask=%r -> last4=%s, account=%s)",
                txn.id,
                txn.account_mask,
                digits,
                txn.account_id,
            )
            return True

    # ---- 4. bank-only fallback ----
    # Only link when the bank has exactly ONE account registered.  If there
    # are multiple accounts (e.g. IndusInd has a savings account AND several
    # credit cards), we must NOT guess -- doing so silently attaches the
    # transaction to the wrong account.  A concrete example: IndusInd CC
    # payment-received emails (email_type='indusind_cc_payment_alert') carry
    # no card_mask and no account_mask because the email body never mentions
    # which CC was credited.  Without this guard those transactions would fall
    # through to the first (savings) account, which is wrong.  Leaving
    # account_id=NULL lets downstream reconciliation (e.g. statement matching
    # by date+amount) attach them correctly later.
    if not txn.card_mask and not txn.account_mask:
        bank_key = txn.bank.strip().lower()
        acct_ids = ctx.accounts_by_bank.get(bank_key, [])
        if len(acct_ids) == 1:
            txn.account_id = acct_ids[0]
            logger.debug(
                "txn %s: linked via bank-only fallback (bank=%r, account=%s)",
                txn.id,
                txn.bank,
                txn.account_id,
            )
            return True
        if len(acct_ids) > 1:
            logger.warning(
                "txn %s: bank-only fallback skipped -- %d accounts for bank %r "
                "(no card_mask / account_mask; leaving unlinked to avoid wrong attribution)",
                txn.id,
                len(acct_ids),
                txn.bank,
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
