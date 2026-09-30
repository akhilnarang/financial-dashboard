"""Statement enrichment replaces a mask, keeps a name, and fills the narration.

An SMS alert for an inbound IMPS names the other party only by a masked mobile
and carries no description at all. The monthly statement is the only source
that holds the real name and the narration, so a matched statement row must be
able to write both onto the existing transaction.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.statements import bank as bank_module
from financial_dashboard.services.statements.bank import enrich_matched_transactions
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio


@pytest.fixture
async def maker(monkeypatch):
    engine, holder = new_test_engine()
    m = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(bank_module, "async_session", m)
    yield m
    await engine.dispose()
    holder.close()


async def _store(maker, *, bank="idfc", **kwargs) -> int:
    async with maker() as s:
        row = Transaction(
            bank=bank,
            email_type="sms",
            direction="credit",
            amount=Decimal("100000.00"),
            currency="INR",
            channel="imps",
            **kwargs,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _read(maker, row_id: int) -> Transaction:
    async with maker() as s:
        return await s.get(Transaction, row_id)


async def _enrich(row_id: int, *, counterparty: str, narration: str) -> int:
    return await enrich_matched_transactions(
        {
            "matched": [
                {
                    "db_txn_id": row_id,
                    "counterparty": counterparty,
                    "narration": narration,
                }
            ]
        }
    )


@pytest.mark.parametrize("masked", ["Mobile XXXXX00006", "payment received", None])
async def test_a_mask_is_replaced_by_the_real_name(maker, masked):
    row_id = await _store(maker, counterparty=masked)

    count = await _enrich(
        row_id,
        counterparty="JANE ROE",
        narration="IMPS/000000000001/JANE ROE/ICIC0000000/0000/Selftransfer",
    )

    stored = await _read(maker, row_id)
    assert count == 1
    assert stored.counterparty == "JANE ROE"
    assert stored.raw_description is not None
    assert "Selftransfer" in stored.raw_description


async def test_a_real_name_is_never_overwritten(maker):
    real_name = "Acct XXXXXXX0003/ALEX DOE"
    row_id = await _store(maker, counterparty=real_name, raw_description="already here")

    await _enrich(row_id, counterparty="SOMEONE ELSE", narration="OTHER NARRATION")

    assert (await _read(maker, row_id)).counterparty == real_name


@pytest.mark.parametrize("method", ["llm", "manual"])
async def test_the_narration_is_filled_even_when_the_name_is_kept(maker, method):
    """The two fields are independent: a real name must not block the narration."""
    from financial_dashboard.services.categorization.hashing import (
        build_input_payload,
        compute_input_hash,
    )

    row_id = await _store(
        maker,
        counterparty="ALICE SMITH",
        category="repayment",
        category_method=method,
        review_status="notified",
    )
    async with maker() as session:
        row = await session.get(Transaction, row_id)
        row.category_input_hash = compute_input_hash(build_input_payload(row, None))
        await session.commit()

    count = await _enrich(
        row_id, counterparty="SOMEONE ELSE", narration="IMPS/1/REMARK/Selftransfer"
    )

    stored = await _read(maker, row_id)
    assert count == 1
    assert stored.counterparty == "ALICE SMITH"
    assert stored.raw_description == "IMPS/1/REMARK/Selftransfer"
    assert stored.category_method == ("pending_llm" if method == "llm" else "manual")
    assert stored.review_status == (None if method == "llm" else "notified")


async def test_an_existing_narration_is_never_overwritten(maker):
    row_id = await _store(
        maker, counterparty="JANE ROE", raw_description="the original"
    )

    count = await _enrich(
        row_id, counterparty="JANE ROE", narration="a different narration"
    )

    assert count == 0
    assert (await _read(maker, row_id)).raw_description == "the original"


async def test_a_narration_alone_does_not_replace_a_mask(maker):
    """A statement row the parser could not resolve names nobody.

    Its narration can be a channel label such as "MOBILE BANKING". Writing that
    over a mask swaps one non-identifying value for another, and the label would
    then look authoritative and block a real name later. The narration still
    reaches the description.
    """
    row_id = await _store(maker, counterparty="Mobile XXXXX00006")

    await _enrich(row_id, counterparty="", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "Mobile XXXXX00006"
    assert stored.raw_description == "MOBILE BANKING"


async def test_a_statement_name_replaces_a_saved_label(maker):
    """A label the user chose is weaker than a name the bank states.

    The statement is the bank speaking, so its name wins, and the source
    follows the name.
    """
    row_id = await _store(
        maker, counterparty="My Saved Payee", counterparty_source="user_alias"
    )

    await _enrich(row_id, counterparty="RAISESEC URITIES", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "RAISESEC URITIES"
    assert stored.counterparty_source == "bank"


async def test_a_statement_that_states_the_saved_label_clears_the_claim(maker):
    """The statement can carry the same text the user saved as a label.

    The name does not change, so nothing is written. The bank still stated
    the text, so the source must stop calling it a label. If it does not, a
    later label replaces a name the statement confirmed.
    """
    row_id = await _store(
        maker, counterparty="RAISESEC URITIES", counterparty_source="user_alias"
    )

    await _enrich(row_id, counterparty="RAISESEC URITIES", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "RAISESEC URITIES"
    assert stored.counterparty_source == "bank"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("Self", "Slice FD"), ("ACME CORP", "ACME CORP")],
)
async def test_an_fd_label_upgrades_only_self(maker, stored, expected):
    """The FD label may come from a narration. It replaces only "Self"."""
    row_id = await _store(maker, counterparty=stored, bank="slice")

    await _enrich(row_id, counterparty="", narration="Slice FD")

    assert (await _read(maker, row_id)).counterparty == expected
