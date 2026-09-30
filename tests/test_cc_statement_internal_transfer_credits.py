"""Bank-internal credit rows on a CC statement must not become transactions.

HSBC prints each billed EMI instalment twice in ``PURCHASES & INSTALLMENTS``:
a ``CR`` row that moves the instalment off the loan ledger, then the debit
that bills it. cc-parser keeps the ``CR`` row in ``payments_refunds`` for
observability and tags it ``credit_reasons="emi_installment_transfer"``. The
reconciler must drop that row; importing it creates a phantom
credit-card-payment credit that mirrors the instalment debit.
"""

from types import SimpleNamespace

import pytest
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db import Account, StatementUpload, Transaction
from tests.conftest import new_test_engine
from financial_dashboard.services.statements import cc as cc_module
from financial_dashboard.services.statements.cc import (
    claim_internal_transfer_twin,
    import_missing_cc_txns,
    internal_transfer_debit_twins,
    load_account_card_masks,
    reconcile_statement,
)

ACCOUNT_ID = 1
CARD_NUMBER = "4000XXXXXXXX0001"
INSTALMENT = "TEST MERCHANT CC000000000000 1ST OF 3 INSTALLMENTS PRINCIPAL"


@pytest.fixture
async def session_factory(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


def _row(direction: str, credit_reasons: str | None = None) -> CcTransaction:
    return CcTransaction(
        date="15/07/2026",
        narration=INSTALMENT,
        amount="300.00",
        card_number=CARD_NUMBER,
        transaction_type=direction,
        credit_reasons=credit_reasons,
    )


def _parsed(debits: list[CcTransaction], credits: list[CcTransaction]):
    return SimpleNamespace(
        bank="hsbc",
        transactions=debits,
        payments_refunds=credits,
        payments_refunds_total="0.00",
        card_summaries=[],
        possible_adjustment_pairs=[],
        overall_total="0.00",
        overall_reward_points="0",
    )


def test_a_twin_on_another_card_does_not_count():
    other_card = CcTransaction(
        date="15/07/2026",
        narration=INSTALMENT,
        amount="300.00",
        card_number="4000XXXXXXXX0002",
        transaction_type="debit",
    )
    twins = internal_transfer_debit_twins([other_card])
    assert not claim_internal_transfer_twin(
        _row("credit", "emi_installment_transfer"), twins
    )


def test_a_twin_alone_does_not_drop_an_untagged_credit():
    twins = internal_transfer_debit_twins([_row("debit")])
    assert not claim_internal_transfer_twin(_row("credit", "cr_marker"), twins)
    assert not claim_internal_transfer_twin(_row("credit"), twins)


@pytest.mark.anyio
async def test_emi_transfer_credit_is_neither_matched_nor_imported(session_factory):
    """A tagged credit with a debit twin is dropped. Each twin is claimed once,
    so a second tagged credit (a reversed instalment) still imports, and so
    does an untagged credit."""
    async with session_factory() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hsbc",
                label="HSBC Credit Card",
                type="credit_card",
                account_number=CARD_NUMBER,
            )
        )
        await session.commit()

    parsed = _parsed(
        debits=[_row("debit")],
        credits=[
            _row("credit", "emi_installment_transfer"),
            _row("credit", "emi_installment_transfer"),
            # A genuine credit on the same day and amount still imports.
            CcTransaction(
                date="15/07/2026",
                narration="BBPS PMT TESTREF",
                amount="300.00",
                card_number=CARD_NUMBER,
                transaction_type="credit",
                credit_reasons="cr_marker",
            ),
        ],
    )

    async with session_factory() as session:
        card_masks = await load_account_card_masks(session, ACCOUNT_ID)
    recon = reconcile_statement(parsed, [], ACCOUNT_ID, card_masks)

    assert recon["matched"] == []
    assert [(e["direction"], e["narration"]) for e in recon["missing"]] == [
        ("debit", INSTALMENT),
        ("credit", INSTALMENT),
        ("credit", "BBPS PMT TESTREF"),
    ]

    async with session_factory() as session:
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hsbc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="parsed",
        )
        session.add(upload)
        await session.flush()
        account = await session.get(Account, ACCOUNT_ID)
        await import_missing_cc_txns(session, upload, parsed, account, recon)
        await session.commit()

    async with session_factory() as session:
        rows = list((await session.execute(select(Transaction))).scalars().all())
    assert sorted((r.direction, r.counterparty) for r in rows) == [
        ("credit", "BBPS PMT TESTREF"),
        ("credit", INSTALMENT),
        ("debit", INSTALMENT),
    ]
