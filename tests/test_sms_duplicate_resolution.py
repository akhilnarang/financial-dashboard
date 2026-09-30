"""Tests for atomic deferred-SMS duplicate resolution."""

import asyncio
import datetime
from decimal import Decimal

import pytest
from bank_sms_parser.models import Money, ParsedSms, SmsTransactionAlert
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, SmsMessage, Transaction
from financial_dashboard.services.sms_duplicate_resolution import (
    SmsDuplicateResolutionError,
    resolve_sms_duplicate,
)
from financial_dashboard.services.txn_merge import DUP_DEFER_NOTE


def _parsed(*, amount="246.80", balance="5753.20", reference=None) -> ParsedSms:
    return ParsedSms(
        email_type="sample_debit_alert",
        bank="samplebank",
        transaction=SmsTransactionAlert(
            direction="debit",
            amount=Money(amount=Decimal(amount), currency="INR"),
            transaction_date=datetime.date(2026, 8, 12),
            transaction_time=datetime.time(10, 15),
            counterparty="Synthetic Shop",
            balance=(
                Money(amount=Decimal(balance), currency="INR")
                if balance is not None
                else None
            ),
            card_mask="4242",
            reference_number=reference,
        ),
    )


def _patch_parsers(monkeypatch, parsed: ParsedSms) -> None:
    monkeypatch.setattr(
        "financial_dashboard.services.sms_duplicate_resolution.parse_sms",
        lambda *args, **kwargs: parsed,
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *args, **kwargs: parsed,
    )


async def _seed_deferred(
    session: AsyncSession,
    *,
    target_amount="246.80",
    target_balance=None,
    reference=None,
) -> tuple[int, int]:
    target = Transaction(
        bank="samplebank",
        email_type="sample_debit_alert",
        direction="debit",
        amount=Decimal(target_amount),
        currency="INR",
        transaction_date=datetime.date(2026, 8, 12),
        transaction_time=datetime.time(10, 15),
        counterparty="Synthetic Shop",
        balance=Decimal(target_balance) if target_balance is not None else None,
        reference_number=reference,
        source="email",
        email_id=42,
    )
    sms = SmsMessage(
        bank="samplebank",
        sender="AD-SAMPLE",
        body="Synthetic duplicate alert",
        received_at=datetime.datetime(2026, 8, 12, 4, 45, tzinfo=datetime.UTC),
        status="skipped",
        parse_error=DUP_DEFER_NOTE,
    )
    session.add_all([target, sms])
    await session.commit()
    return sms.id, target.id


@pytest.mark.anyio
async def test_resolver_merges_without_stealing_transaction_sms_slot(
    session, monkeypatch
):
    parsed = _parsed()
    _patch_parsers(monkeypatch, parsed)
    sms_id, target_id = await _seed_deferred(session)

    result = await resolve_sms_duplicate(session, sms_id, "merge", target_id)

    assert result.status == "merged"
    assert result.transaction_id == target_id
    sms = await session.get(SmsMessage, sms_id)
    target = await session.get(Transaction, target_id)
    assert sms.transaction_id == target_id
    assert sms.status == "enriched"
    assert sms.parse_error is None
    assert target.sms_message_id is None
    assert target.balance == Decimal("5753.20")
    assert target.card_mask == "4242"


@pytest.mark.anyio
async def test_resolver_creates_new_transaction(session, monkeypatch):
    parsed = _parsed()
    _patch_parsers(monkeypatch, parsed)
    sms_id, target_id = await _seed_deferred(session)

    result = await resolve_sms_duplicate(session, sms_id, "create_new")

    assert result.status == "created"
    assert result.transaction_id != target_id
    sms = await session.get(SmsMessage, sms_id)
    created = await session.get(Transaction, result.transaction_id)
    assert sms.transaction_id == created.id
    assert created.sms_message_id == sms_id
    rows = (await session.scalars(select(Transaction))).all()
    assert len(rows) == 2
    # The new row gets its own notification, so its note and category can be set.
    assert result.notification is not None
    assert result.notification["amount"] == created.amount


@pytest.mark.anyio
async def test_resolver_reports_already_resolved_from_reverse_link(
    session, monkeypatch
):
    _patch_parsers(monkeypatch, _parsed())
    sms_id, target_id = await _seed_deferred(session)
    target = await session.get(Transaction, target_id)
    target.sms_message_id = sms_id
    await session.commit()

    result = await resolve_sms_duplicate(session, sms_id, "merge", target_id)

    assert result.status == "already_resolved"
    assert result.transaction_id == target_id


@pytest.mark.anyio
async def test_resolver_rejects_target_drift(session, monkeypatch):
    _patch_parsers(monkeypatch, _parsed())
    sms_id, target_id = await _seed_deferred(session)
    # A row with the SMS balance appears after the defer. The matcher now
    # picks it, but the target itself stays compatible.
    session.add(
        Transaction(
            bank="samplebank",
            email_type="sample_debit_alert",
            direction="debit",
            amount=Decimal("246.80"),
            currency="INR",
            transaction_date=datetime.date(2026, 8, 12),
            transaction_time=datetime.time(10, 15),
            counterparty="Synthetic Shop",
            balance=Decimal("5753.20"),
            source="email",
            email_id=43,
        )
    )
    await session.commit()

    with pytest.raises(SmsDuplicateResolutionError, match="no longer a compatible"):
        await resolve_sms_duplicate(session, sms_id, "merge", target_id)

    sms = await session.get(SmsMessage, sms_id)
    assert sms.transaction_id is None


@pytest.mark.anyio
async def test_resolver_rejects_create_for_reference_mismatch(session, monkeypatch):
    _patch_parsers(
        monkeypatch,
        _parsed(amount="246.80", balance=None, reference="SYNTH-REF-84"),
    )
    sms_id, _ = await _seed_deferred(
        session,
        target_amount="135.70",
        reference="SYNTH-REF-84",
    )

    with pytest.raises(SmsDuplicateResolutionError, match="cannot create"):
        await resolve_sms_duplicate(session, sms_id, "create_new")

    rows = (await session.scalars(select(Transaction))).all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_double_tap_serializes_to_one_resolution(tmp_path, monkeypatch):
    db_path = tmp_path / "duplicate-resolution.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _patch_parsers(monkeypatch, _parsed())

    async with maker() as seed_session:
        sms_id, target_id = await _seed_deferred(seed_session)

    async def resolve_once():
        async with maker() as callback_session:
            return await resolve_sms_duplicate(
                callback_session, sms_id, "merge", target_id
            )

    first, second = await asyncio.gather(resolve_once(), resolve_once())

    assert {first.status, second.status} == {"merged", "already_resolved"}
    assert first.transaction_id == second.transaction_id == target_id
    async with maker() as check_session:
        sms = await check_session.get(SmsMessage, sms_id)
        assert sms.transaction_id == target_id
    await engine.dispose()
