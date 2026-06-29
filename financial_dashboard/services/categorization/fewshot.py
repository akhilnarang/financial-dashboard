"""Retrieve similar already-categorized transactions as LLM few-shot examples."""

from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.normalize import normalize_counterparty


class FewShotExample(NamedTuple):
    """One past categorization shown to the LLM as guidance.

    A trimmed projection of a Transaction — only the fields the prompt renders.
    ``category`` is the human/rule-assigned label we want the model to imitate.
    """

    counterparty: str | None
    raw_description: str | None
    direction: str
    channel: str | None
    category: str


async def get_similar_examples(
    session: AsyncSession,
    *,
    counterparty: str | None,
    direction: str,
    limit: int = 5,
) -> list[FewShotExample]:
    """Find up to ``limit`` past transactions resembling this one, for few-shot priming.

    Selection: same ``direction``, already categorized by a trusted source
    (``manual`` or ``rule`` — never another LLM guess, to avoid compounding
    errors), most recent first. From that window we keep rows whose normalized
    counterparty overlaps the target's (substring either way), since the same
    merchant recurs under slightly different raw narrations.

    Matching happens in Python because there is no normalized-counterparty column
    to query on; we cap the DB scan at a recent 200-row window so this stays
    cheap. Returns ``[]`` when ``counterparty`` normalizes to empty (nothing to
    match against).
    """
    target = normalize_counterparty(counterparty)
    if not target:
        return []
    # Pull a recent window of categorized, same-direction rows, then filter by
    # normalized-counterparty containment in Python (no normalized column in DB).
    stmt = (
        select(Transaction)
        .where(
            Transaction.direction == direction,
            Transaction.category.is_not(None),
            Transaction.category_method.in_(("manual", "rule")),
        )
        .order_by(Transaction.created_at.desc())
        .limit(200)
    )
    rows = (await session.execute(stmt)).scalars().all()
    out: list[FewShotExample] = []
    for r in rows:
        cand = normalize_counterparty(r.counterparty)
        if cand and (cand in target or target in cand):
            out.append(
                FewShotExample(
                    counterparty=r.counterparty,
                    raw_description=r.raw_description,
                    direction=r.direction,
                    channel=r.channel,
                    category=r.category or "",  # query filters category.is_not(None)
                )
            )
        if len(out) >= limit:
            break
    return out
