"""Reparse enrichment migrates a stored counterparty to a "<Bank> FD" label.

An FD row first imported under an older value (e.g. "Self") must pick up the new
"<Bank> FD" label on re-parse — that label is what protects it from the
reference-pair self-transfer rule. An ordinary non-generic counterparty stays
authoritative.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import Transaction
from tests.conftest import new_test_engine
from financial_dashboard.services.statements import bank as bank_module
from financial_dashboard.services.statements.bank import enrich_matched_transactions

pytestmark = pytest.mark.anyio


@pytest.fixture
async def maker(monkeypatch):
    engine, holder = new_test_engine()
    m = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(bank_module, "async_session", m)
    yield m
    await engine.dispose()
    holder.close()


def _fd_row(counterparty: str) -> Transaction:
    return Transaction(
        bank="slice",
        email_type="bank_statement",
        direction="debit",
        amount=Decimal("50000.00"),
        currency="INR",
        counterparty=counterparty,
        channel="bank_statement",
    )


async def test_reparse_upgrades_self_to_fd_label(maker):
    async with maker() as s:
        row = _fd_row("Self")
        s.add(row)
        await s.commit()
        row_id = row.id

    enriched = await enrich_matched_transactions(
        {
            "matched": [
                {
                    "db_txn_id": row_id,
                    "counterparty": "Slice FD",
                    "narration": "Deposit",
                }
            ]
        }
    )

    assert enriched == 1
    async with maker() as s:
        assert (await s.get(Transaction, row_id)).counterparty == "Slice FD"


async def test_reparse_fd_label_does_not_overwrite_real_name(maker):
    # A real, non-generic counterparty is authoritative. Even if reconciliation
    # matches it to an FD-labeled parse (same date/amount, no reference), the FD
    # upgrade must NOT overwrite the real name — only the "Self" placeholder.
    async with maker() as s:
        row = _fd_row("ACME CORP")
        s.add(row)
        await s.commit()
        row_id = row.id

    await enrich_matched_transactions(
        {
            "matched": [
                {
                    "db_txn_id": row_id,
                    "counterparty": "Slice FD",
                    "narration": "Deposit",
                }
            ]
        }
    )

    # The count is not the subject: the row still gains the narration it had no
    # other source for. What must hold is that the real name survives.
    async with maker() as s:
        assert (await s.get(Transaction, row_id)).counterparty == "ACME CORP"
