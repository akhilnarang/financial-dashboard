"""Find the spends that family members make on the owner's add-on cards."""

from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import mask_matches, normalize_mask
from financial_dashboard.db.models import Card, Transaction
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.slugs import FAMILY_SLUG
from financial_dashboard.services.settings import get_setting


class FamilyCards(NamedTuple):
    """The cards and the card holder names that identify family spends."""

    cards: tuple[Card, ...]
    holders: frozenset[str]


def _holder_key(name: str) -> str:
    return " ".join(name.split()).casefold()


async def load_family_cards(session: AsyncSession) -> FamilyCards:
    """Load the registered cards and the family card holder names.

    Args:
        session: Open async session. Only read.

    Returns:
        Every card, so that a mask can match one card only, and the holder
        names from the ``categorization.family_card_holders`` setting.
    """
    raw = get_setting("categorization.family_card_holders") or ""
    return FamilyCards(
        cards=tuple((await session.scalars(select(Card))).all()),
        holders=frozenset(_holder_key(name) for name in raw.split(",") if name.strip()),
    )


def is_family_spend(family: FamilyCards, txn: Transaction) -> bool:
    """Tell if a row is on a family add-on card.

    A statement prints the holder name. An alert prints only the card mask.
    Some issuers give the add-on the same mask as the primary card. Thus a
    family holder name is enough, and a mask is enough when it identifies one
    family card on the row's account.

    Args:
        family: The cards and holder names from ``load_family_cards``.
        txn: The row to check.

    Returns:
        True when the holder is a family holder, or the row's card is a
        family card.
    """
    if txn.card_holder and _holder_key(txn.card_holder) in family.holders:
        return True
    if txn.card_id is not None:
        matched = [card for card in family.cards if card.id == txn.card_id]
    else:
        incoming = normalize_mask(txn.card_mask)
        matched = [
            card
            for card in family.cards
            if incoming
            and card.account_id == txn.account_id
            and mask_matches(incoming, normalize_mask(card.card_mask))
        ]
    return len(matched) == 1 and matched[0].is_family


async def requeue_family_spend(
    session: AsyncSession, family: FamilyCards, txn: Transaction
) -> bool:
    """Send a family row that is not yet ``family`` back to the rule pass.

    A manual category is the owner's choice, so it stays. A row that the
    family rule does not take, such as a bill payment, keeps its category.

    Args:
        session: Open async session that owns ``txn``.
        family: The cards and holder names from ``load_family_cards``.
        txn: The row to check.

    Returns:
        True when the row goes back to the rule pass.
    """
    if (
        txn.category_method == "manual"
        or txn.category == FAMILY_SLUG
        or not is_family_spend(family, txn)
    ):
        return False
    probe = match_rules(
        {
            "counterparty": txn.counterparty,
            "raw_description": txn.raw_description,
            "direction": txn.direction,
            "email_type": txn.email_type,
            "family_card": True,
        },
        default_rule_config(),
    )
    if probe is None or probe.slug != FAMILY_SLUG:
        return False
    txn.category_method = None
    txn.review_status = None
    txn.review_reason = None
    await supersede_active_decisions(session, txn.id)
    return True
