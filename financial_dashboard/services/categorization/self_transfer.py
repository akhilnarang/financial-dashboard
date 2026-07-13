"""Authoritative self-transfer detection from shared transaction references."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import mask_digits
from financial_dashboard.db.models import Account, Transaction, utc_now
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_input_hash,
)
from financial_dashboard.services.categorization.vocabulary import get_vocab_version

REFERENCE_PAIR_RULESET_VERSION = "reference-pair-v1"

_OPPOSITE_DIRECTION = {
    "credit": "debit",
    "debit": "credit",
}

_MIN_ACCOUNT_MASK_DIGITS = 3


async def _account_type(session: AsyncSession, txn: Transaction) -> str | None:
    if txn.account_id is None:
        return None
    return await session.scalar(
        select(Account.type).where(Account.id == txn.account_id)
    )


async def _mark_self_transfer(session: AsyncSession, txn: Transaction) -> None:
    input_hash = compute_input_hash(
        build_input_payload(txn, await _account_type(session, txn))
    )
    already_current = (
        txn.category == "self_transfer"
        and txn.category_method == "rule"
        and txn.category_confidence == 1.0
        and txn.category_model == REFERENCE_PAIR_RULESET_VERSION
        and txn.category_input_hash == input_hash
        and txn.category_vocab_version == get_vocab_version()
        and txn.review_status is None
        and txn.review_reason is None
    )
    if already_current:
        return

    txn.category = "self_transfer"
    txn.category_method = "rule"
    txn.category_confidence = 1.0
    txn.category_model = REFERENCE_PAIR_RULESET_VERSION
    txn.category_input_hash = input_hash
    txn.category_vocab_version = get_vocab_version()
    txn.categorized_at = utc_now()
    txn.review_status = None
    txn.review_reason = None


def _different_accounts(first: Transaction, second: Transaction) -> bool:
    """Return whether the pair is known to belong to different accounts.

    Linked account IDs are authoritative when both are available. During
    ingest the new row is categorized before the linker runs, so otherwise
    compare its parser-provided account mask. Mask comparison is suffix-aware
    because banks expose anywhere from three trailing digits to a longer
    masked account number. Fewer than three digits are not enough evidence,
    matching the linker's minimum account-suffix policy.

    Missing evidence is treated conservatively: if neither IDs nor masks prove
    that the accounts differ, the rule does not fire.
    """
    if first.account_id is not None and second.account_id is not None:
        return first.account_id != second.account_id

    first_digits = mask_digits(first.account_mask)
    second_digits = mask_digits(second.account_mask)
    if (
        len(first_digits) < _MIN_ACCOUNT_MASK_DIGITS
        or len(second_digits) < _MIN_ACCOUNT_MASK_DIGITS
    ):
        return False
    same_suffix = first_digits.endswith(second_digits) or second_digits.endswith(
        first_digits
    )
    return not same_suffix


async def apply_reference_self_transfer_rule(
    session: AsyncSession, txn: Transaction
) -> bool:
    """Mark both legs when ``txn`` shares a ref with the opposite direction.

    Exact, nonblank references are used deliberately: an IMPS RRN is shared by
    the sending and receiving alerts, while fuzzy normalization would weaken
    the signal. The legs must also belong to different accounts so a merchant
    charge and its same-reference refund are not mistaken for a self-transfer.
    The rule is authoritative and therefore replaces prior LLM, review, or
    manual categorization on both qualifying legs.

    Returns whether at least one opposite-direction leg was found.
    """
    reference_number = txn.reference_number
    opposite = _OPPOSITE_DIRECTION.get(txn.direction)
    if not reference_number or not reference_number.strip() or opposite is None:
        return False

    stmt = select(Transaction).where(
        Transaction.reference_number == reference_number,
        Transaction.direction == opposite,
    )
    matches = [
        match
        for match in (await session.scalars(stmt)).all()
        if _different_accounts(txn, match)
    ]
    if not matches:
        return False

    await _mark_self_transfer(session, txn)
    for match in matches:
        await _mark_self_transfer(session, match)
    return True
