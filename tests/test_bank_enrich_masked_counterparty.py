"""Statement enrichment replaces a mask, keeps a name, and fills the narration.

An SMS alert for an inbound IMPS names the other party only by a masked mobile
and carries no description at all. The monthly statement is the only source
that holds the real name and the narration, so a matched statement row must be
able to write both onto the existing transaction.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.models import Base, Transaction
from financial_dashboard.services.statements import bank as bank_module
from financial_dashboard.services.statements.bank import enrich_matched_transactions

pytestmark = pytest.mark.anyio


@pytest.fixture
async def maker(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    m = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(bank_module, "async_session", m)
    yield m
    await engine.dispose()


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


@pytest.mark.parametrize(
    "masked",
    [
        "Mobile XXXXXXXXX006",
        "Mobile XXXXX64006",
        "Acct XXXXXXXXX214",
        "Acct xxxxxxxxxx8214",
        "xxxxxxxxxx8669",
        "payment received",
    ],
)
async def test_a_mask_is_replaced_by_the_real_name(maker, masked):
    row_id = await _store(maker, counterparty=masked)

    count = await _enrich(
        row_id,
        counterparty="ANJALIJY OTESHNA",
        narration="IMPS/613111314099/ANJALIJY OTESHNA/ICIC0000004/0424/Selftransfer",
    )

    assert count == 1
    assert (await _read(maker, row_id)).counterparty == "ANJALIJY OTESHNA"


@pytest.mark.parametrize(
    "real_name",
    [
        "ANJALIJY OTESHNA",
        "ZEPTO MARKETPLACE",
        "Acct XXXXXXX7703/AKHIL JYOT",
        # An X-run with no digits is not a mask. Nothing says these name nobody,
        # so they must be kept.
        "XXX",
        "XXXX",
        "Mobile XXX",
    ],
)
async def test_a_real_name_is_never_overwritten(maker, real_name):
    row_id = await _store(maker, counterparty=real_name, raw_description="already here")

    await _enrich(row_id, counterparty="SOMEONE ELSE", narration="OTHER NARRATION")

    assert (await _read(maker, row_id)).counterparty == real_name


async def test_a_row_with_no_description_gains_the_narration(maker):
    """An SMS row carries no description. The statement is the only source."""
    row_id = await _store(maker, counterparty="Mobile XXXXXXXXX006")
    assert (await _read(maker, row_id)).raw_description is None

    await _enrich(
        row_id,
        counterparty="ANJALIJY OTESHNA",
        narration="IMPS/613111314099/ANJALIJY OTESHNA/ICIC0000004/0424/Selftransfer",
    )

    stored = await _read(maker, row_id)
    assert stored.raw_description is not None
    assert "Selftransfer" in stored.raw_description


async def test_the_narration_is_filled_even_when_the_name_is_kept(maker):
    """The two fields are independent: a real name must not block the narration."""
    row_id = await _store(maker, counterparty="ANJALIJY OTESHNA")

    count = await _enrich(
        row_id, counterparty="SOMEONE ELSE", narration="IMPS/1/REMARK/Selftransfer"
    )

    stored = await _read(maker, row_id)
    assert count == 1
    assert stored.counterparty == "ANJALIJY OTESHNA"
    assert stored.raw_description == "IMPS/1/REMARK/Selftransfer"


async def test_an_existing_narration_is_never_overwritten(maker):
    row_id = await _store(
        maker, counterparty="ANJALIJY OTESHNA", raw_description="the original"
    )

    count = await _enrich(
        row_id, counterparty="ANJALIJY OTESHNA", narration="a different narration"
    )

    assert count == 0
    assert (await _read(maker, row_id)).raw_description == "the original"


async def test_an_unchanged_value_is_not_reported_as_enriched(maker):
    """A reparse that yields what is already stored has enriched nothing."""
    row_id = await _store(
        maker, counterparty="ANJALIJY OTESHNA", raw_description="IMPS/1/SAME"
    )

    count = await _enrich(
        row_id, counterparty="ANJALIJY OTESHNA", narration="IMPS/1/SAME"
    )

    assert count == 0


async def test_a_mask_replaced_by_the_identical_mask_is_not_enriched(maker):
    """The statement can repeat the mask. Repeating it changes nothing."""
    row_id = await _store(
        maker, counterparty="Mobile XXXXXXXXX006", raw_description="already here"
    )

    count = await _enrich(
        row_id, counterparty="Mobile XXXXXXXXX006", narration="already here"
    )

    assert count == 0
    assert (await _read(maker, row_id)).counterparty == "Mobile XXXXXXXXX006"


async def test_a_narration_alone_does_not_replace_a_mask(maker):
    """A statement row the parser could not resolve names nobody.

    Its narration can be a channel label such as "MOBILE BANKING". Writing that
    over a mask swaps one non-identifying value for another, and the label would
    then look authoritative and block a real name later. The narration still
    reaches the description.
    """
    row_id = await _store(maker, counterparty="Mobile XXXXX64006")

    await _enrich(row_id, counterparty="", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "Mobile XXXXX64006"
    assert stored.raw_description == "MOBILE BANKING"


async def test_a_parsed_counterparty_replaces_a_mask(maker):
    """The parser resolving a party is what makes the value trustworthy."""
    row_id = await _store(maker, counterparty="Mobile XXXXX64006")

    await _enrich(row_id, counterparty="RAISESEC URITIES", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "RAISESEC URITIES"
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


async def test_a_narration_alone_does_not_fill_an_empty_counterparty(maker):
    """An empty field is not a licence to store a channel label.

    Writing one would make it look authoritative, so a real name arriving later
    could never replace it. The narration still reaches the description.
    """
    row_id = await _store(maker, counterparty=None)

    await _enrich(row_id, counterparty="", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty is None
    assert stored.raw_description == "MOBILE BANKING"


async def test_a_narration_alone_does_not_replace_a_generic_placeholder(maker):
    row_id = await _store(maker, counterparty="payment received")

    await _enrich(row_id, counterparty="", narration="MOBILE BANKING")

    stored = await _read(maker, row_id)
    assert stored.counterparty == "payment received"
    assert stored.raw_description == "MOBILE BANKING"


async def test_a_parsed_counterparty_fills_an_empty_field(maker):
    row_id = await _store(maker, counterparty=None)

    await _enrich(row_id, counterparty="ANJALIJY OTESHNA", narration="IMPS/1/X")

    assert (await _read(maker, row_id)).counterparty == "ANJALIJY OTESHNA"


async def test_a_narration_derived_fd_label_still_upgrades_self(maker):
    """The FD upgrade is the one exception, and it may come from a narration."""
    row_id = await _store(maker, counterparty="Self", bank="slice")

    await _enrich(row_id, counterparty="", narration="Slice FD")

    assert (await _read(maker, row_id)).counterparty == "Slice FD"
