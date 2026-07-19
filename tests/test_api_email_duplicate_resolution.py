"""Explicit resolution of deferred email duplicates."""

import asyncio
import datetime
import logging
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, Email, FetchRule, Transaction
from financial_dashboard.integrations.email.body import RawEmailLoadResult
from financial_dashboard.schemas.emails import DuplicateResolutionRequest
from financial_dashboard.services.emails import ProcessedEmailParse
from financial_dashboard.services.txn_merge import DUP_DEFER_NOTE


RAW_EMAIL = b"synthetic-rfc822-message"


def _parsed_txn(**changes) -> dict:
    values = {
        "bank": "testbank",
        "email_type": "testbank_debit_alert",
        "direction": "debit",
        "amount": Decimal("42.15"),
        "currency": "inr",
        "transaction_date": datetime.date(2030, 1, 2),
        "transaction_time": datetime.time(10, 4),
        "counterparty": "Synthetic Shop",
        "card_mask": None,
        "account_mask": "XX0123",
        "reference_number": "FULLREF0123",
        "channel": "online",
        "balance": Decimal("900.00"),
        "raw_description": "Synthetic purchase description",
    }
    values.update(changes)
    return values


async def _seed_deferred(session, *, target_email_id=None) -> tuple[Email, Transaction]:
    rule = FetchRule(
        provider="gmail",
        sender="alerts@example.invalid",
        bank="testbank",
        email_kind="transaction",
        enabled=True,
    )
    session.add(rule)
    await session.flush()
    email = Email(
        provider="gmail",
        message_id="synthetic-deferred-message",
        sender="alerts@example.invalid",
        subject="Synthetic debit alert",
        status="skipped",
        error=DUP_DEFER_NOTE,
        rule_id=rule.id,
    )
    session.add(email)
    await session.flush()
    target = Transaction(
        email_id=target_email_id,
        bank="testbank",
        email_type="testbank_sms_debit_alert",
        direction="debit",
        amount=Decimal("42.15"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 5),
        counterparty=None,
        account_mask=None,
        reference_number="0123",
        channel="upi",
        balance=None,
        raw_description=None,
        source="sms",
        notified_channel="sms",
        category="synthetic_category",
        category_method="manual",
        note="keep this note",
    )
    session.add(target)
    await session.commit()
    return email, target


def _patch_current_parse(monkeypatch, txn_data: dict | None = None) -> AsyncMock:
    from financial_dashboard.services import duplicate_resolution as service

    loader = AsyncMock(return_value=RawEmailLoadResult(RAW_EMAIL, None))
    monkeypatch.setattr(service, "load_or_fetch_raw_email", loader)
    result = ProcessedEmailParse(
        None if txn_data is not None else "synthetic parse failure",
        txn_data,
        None,
        None,
    )
    monkeypatch.setattr(service, "_process_email_full", lambda _bank, _raw: result)
    return loader


@pytest.mark.anyio
@pytest.mark.parametrize(
    "preview_token",
    [
        pytest.param("v1." + "a" * 63 + "é", id="non-ascii"),
        pytest.param("not-a-preview-token", id="wrong-shape"),
        pytest.param("v1." + "A" * 64, id="uppercase-hex"),
        pytest.param("v1." + "a" * 63, id="truncated"),
    ],
)
async def test_apply_rejects_invalid_preview_token_before_service(
    client, session, monkeypatch, preview_token
):
    email, target = await _seed_deferred(session)
    email_id, target_id = email.id, target.id
    import financial_dashboard.api.emails as emails_api

    resolver = AsyncMock()
    monkeypatch.setattr(emails_api, "resolve_email_duplicate", resolver)

    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={
            "transaction_id": target_id,
            "apply": True,
            "preview_token": preview_token,
        },
    )

    assert response.status_code == 422, response.text
    resolver.assert_not_awaited()
    session.expire_all()
    assert (await session.get(Email, email_id)).status == "skipped"
    assert (await session.get(Transaction, target_id)).email_id is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"apply": True}, id="apply-requires-token"),
        pytest.param(
            {"preview_token": "v1." + "a" * 64},
            id="preview-forbids-token",
        ),
    ],
)
async def test_request_mode_rejects_missing_or_forbidden_token_before_service(
    client, session, monkeypatch, payload
):
    email, target = await _seed_deferred(session)
    import financial_dashboard.api.emails as emails_api

    resolver = AsyncMock()
    monkeypatch.setattr(emails_api, "resolve_email_duplicate", resolver)

    response = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id, **payload},
    )

    assert response.status_code == 422, response.text
    resolver.assert_not_awaited()


@pytest.mark.anyio
async def test_preview_is_default_and_has_no_side_effects(client, session, monkeypatch):
    email, target = await _seed_deferred(session)
    email_id, target_id = email.id, target.id
    _patch_current_parse(monkeypatch, _parsed_txn())
    original_fetched_at = email.fetched_at
    original_created_at = target.created_at
    assert original_fetched_at is not None
    assert original_created_at is not None

    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "preview"
    assert body["email_status"] == "skipped"
    assert body["preview_token"].startswith("v1.")
    assert set(body["diff"]["changed_fields"]) == {
        "transaction_time",
        "counterparty",
        "account_mask",
        "reference_number",
        "channel",
        "balance",
        "raw_description",
    }
    assert body["before"]["email_id"] is None
    assert body["after"]["email_id"] == email_id
    assert body["after"]["source"] == "sms+email"
    assert body["after"]["reference_number"] == "FULLREF0123"

    session.expire_all()
    stored_email = await session.get(Email, email_id)
    stored_target = await session.get(Transaction, target_id)
    assert stored_email.status == "skipped"
    assert stored_email.error == DUP_DEFER_NOTE
    assert stored_email.fetched_at == original_fetched_at.replace(tzinfo=None)
    assert stored_target.email_id is None
    assert stored_target.source == "sms"
    assert stored_target.enriched_at is None
    assert stored_target.created_at == original_created_at.replace(tzinfo=None)
    assert stored_target.counterparty is None


@pytest.mark.anyio
async def test_apply_enriches_existing_row_atomically_without_payment_or_duplicate(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    account = Account(
        bank="testbank",
        label="Synthetic checking",
        type="bank_account",
        account_number="000000000123",
    )
    session.add(account)
    await session.commit()
    email_id, target_id, account_id = email.id, target.id, account.id
    _patch_current_parse(monkeypatch, _parsed_txn())

    from financial_dashboard.services import reminders, telegram

    payment_check = AsyncMock()
    notification = AsyncMock()
    monkeypatch.setattr(reminders, "check_payment_received", payment_check)
    monkeypatch.setattr(telegram, "send_transaction_notification", notification)

    preview = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )
    token = preview.json()["preview_token"]
    applied = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={
            "transaction_id": target_id,
            "apply": True,
            "preview_token": token,
        },
    )

    assert applied.status_code == 200, applied.text
    assert applied.json()["mode"] == "applied"
    assert applied.json()["after"]["reference_number"] == "FULLREF0123"
    session.expire_all()
    stored_email = await session.get(Email, email_id)
    stored_target = await session.get(Transaction, target_id)
    txn_count = await session.scalar(select(func.count()).select_from(Transaction))
    assert txn_count == 1
    assert stored_email.status == "parsed"
    assert stored_email.error is None
    assert stored_target.email_id == email_id
    assert stored_target.source == "sms+email"
    assert stored_target.reference_number == "FULLREF0123"
    assert stored_target.counterparty == "Synthetic Shop"
    assert stored_target.account_id == account_id
    assert stored_target.enriched_at is not None
    assert stored_target.category == "synthetic_category"
    assert stored_target.category_method == "manual"
    assert stored_target.note == "keep this note"
    payment_check.assert_not_awaited()
    notification.assert_not_awaited()


@pytest.mark.anyio
async def test_apply_relinks_after_richer_mask_overwrites_unusable_mask(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    target.account_mask = "XX23"
    account = Account(
        bank="testbank",
        label="Synthetic checking",
        type="bank_account",
        account_number="000000000123",
    )
    session.add(account)
    await session.commit()
    email_id, target_id, account_id = email.id, target.id, account.id
    _patch_current_parse(monkeypatch, _parsed_txn(account_mask="XX0123"))

    preview = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["diff"]["overwritten"] == [
        "transaction_time",
        "account_mask",
        "reference_number",
        "channel",
    ]
    applied = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={
            "transaction_id": target_id,
            "apply": True,
            "preview_token": preview.json()["preview_token"],
        },
    )

    assert applied.status_code == 200, applied.text
    session.expire_all()
    stored = await session.get(Transaction, target_id)
    assert stored.account_mask == "XX0123"
    assert stored.account_id == account_id


@pytest.mark.anyio
async def test_apply_rejects_stale_preview_token(client, session, monkeypatch):
    email, target = await _seed_deferred(session)
    email_id, target_id = email.id, target.id
    _patch_current_parse(monkeypatch, _parsed_txn())
    preview = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )
    token = preview.json()["preview_token"]

    stored_target = await session.get(Transaction, target_id)
    stored_target.counterparty = "Changed after preview"
    await session.commit()
    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={
            "transaction_id": target_id,
            "apply": True,
            "preview_token": token,
        },
    )

    assert response.status_code == 409
    session.expire_all()
    assert (await session.get(Email, email_id)).status == "skipped"
    assert (await session.get(Transaction, target_id)).email_id is None


@pytest.mark.anyio
async def test_incompatible_target_is_rejected(client, session, monkeypatch):
    email, target = await _seed_deferred(session)
    _patch_current_parse(monkeypatch, _parsed_txn(amount=Decimal("43.15")))

    response = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert response.status_code == 409
    assert "compatible" in response.json()["detail"]


@pytest.mark.anyio
async def test_conflicting_known_balance_is_incompatible(client, session, monkeypatch):
    email, target = await _seed_deferred(session)
    target.balance = Decimal("850.00")
    await session.commit()
    email_id, target_id = email.id, target.id
    _patch_current_parse(monkeypatch, _parsed_txn(balance=Decimal("900.00")))

    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )

    assert response.status_code == 409
    assert "compatible" in response.json()["detail"]


@pytest.mark.anyio
async def test_target_with_occupied_email_slot_is_rejected(
    client, session, monkeypatch
):
    occupied = Email(
        provider="gmail",
        message_id="synthetic-occupied-message",
        status="parsed",
    )
    session.add(occupied)
    await session.flush()
    email, target = await _seed_deferred(session, target_email_id=occupied.id)
    loader = _patch_current_parse(monkeypatch, _parsed_txn())

    response = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert response.status_code == 409
    assert "already has an email" in response.json()["detail"]
    loader.assert_not_awaited()


@pytest.mark.anyio
async def test_source_with_attached_transaction_is_rejected(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    attached = Transaction(
        email_id=email.id,
        bank="other-test-bank",
        email_type="synthetic_attached_alert",
        direction="credit",
        amount=Decimal("7.25"),
    )
    session.add(attached)
    await session.commit()
    email_id, target_id = email.id, target.id
    loader = _patch_current_parse(monkeypatch, _parsed_txn())

    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )

    assert response.status_code == 409
    assert "Email already has an attached" in response.json()["detail"]
    loader.assert_not_awaited()


@pytest.mark.anyio
async def test_non_deferred_email_is_rejected(client, session, monkeypatch):
    email, target = await _seed_deferred(session)
    email.status = "failed"
    email.error = "synthetic prior parse error"
    await session.commit()
    loader = _patch_current_parse(monkeypatch, _parsed_txn())

    response = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert response.status_code == 409
    loader.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("rule_state", "expected_status"),
    [("missing", 404), ("non_transaction", 422)],
)
async def test_ineligible_rule_does_not_load_raw_email(
    client, session, monkeypatch, rule_state, expected_status
):
    email, target = await _seed_deferred(session)
    rule = await session.get(FetchRule, email.rule_id)
    assert rule is not None
    if rule_state == "missing":
        await session.delete(rule)
    else:
        rule.email_kind = "cc_statement"
    await session.commit()
    loader = _patch_current_parse(monkeypatch, _parsed_txn())

    response = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert response.status_code == expected_status
    loader.assert_not_awaited()


@pytest.mark.anyio
async def test_current_parse_failure_is_422_and_keeps_defer(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    email_id, target_id = email.id, target.id
    _patch_current_parse(monkeypatch, None)

    response = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Current parser did not produce a transaction"
    session.expire_all()
    assert (await session.get(Email, email_id)).status == "skipped"


@pytest.mark.anyio
async def test_missing_email_transaction_and_raw_source_return_404(
    client, session, monkeypatch, caplog
):
    email, target = await _seed_deferred(session)
    email_id, target_id = email.id, target.id
    loader = _patch_current_parse(monkeypatch, _parsed_txn())

    missing_email = await client.post(
        "/api/emails/999999/resolve-duplicate",
        json={"transaction_id": target_id},
    )
    missing_txn = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": 999999},
    )
    loader.assert_not_awaited()
    from financial_dashboard.services import duplicate_resolution as service

    monkeypatch.setattr(
        service,
        "load_or_fetch_raw_email",
        AsyncMock(return_value=RawEmailLoadResult(None, "secret provider detail")),
    )
    with caplog.at_level(logging.WARNING, logger=service.__name__):
        missing_raw = await client.post(
            f"/api/emails/{email_id}/resolve-duplicate",
            json={"transaction_id": target_id},
        )

    assert missing_email.status_code == 404
    assert missing_txn.status_code == 404
    assert missing_raw.status_code == 404
    assert missing_raw.json()["detail"] == "Raw email source is unavailable"
    assert "secret" not in missing_raw.text
    assert any(
        str(email_id) in record.getMessage()
        and "secret provider detail" in record.getMessage()
        for record in caplog.records
        if record.name == service.__name__
    )


@pytest.mark.anyio
async def test_single_reparse_parse_failure_refreshes_error_without_race(
    client, session, monkeypatch
):
    email, _target = await _seed_deferred(session)
    email.status = "failed"
    email.error = "stale parser detail"
    await session.commit()
    email_id = email.id

    from financial_dashboard.web import emails as emails_web

    loader = AsyncMock(return_value=RawEmailLoadResult(RAW_EMAIL, None))
    parser = AsyncMock(return_value=("current parser detail", None, None, None))
    monkeypatch.setattr(emails_web, "load_or_fetch_raw_email", loader)
    monkeypatch.setattr(emails_web, "parse_email_by_kind", parser)
    monkeypatch.setattr(emails_web, "_save_failed_email", lambda *_args: None)

    response = await client.post(f"/emails/{email_id}/reparse")

    assert response.status_code == 422
    assert response.json()["detail"] == "current parser detail"
    loader.assert_awaited_once()
    parser.assert_awaited_once()
    session.expire_all()
    stored = await session.get(Email, email_id)
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "current parser detail"


@pytest.mark.anyio
async def test_reparse_parse_failure_does_not_overwrite_concurrent_explicit_apply(
    tmp_path, monkeypatch
):
    """A stale parser result must not restore an error after explicit apply."""
    from financial_dashboard.services import duplicate_resolution as service
    from financial_dashboard.web import emails as emails_web

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'reparse-parse-failure-race.sqlite'}",
        connect_args={"timeout": 5},
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with maker() as seed_session:
            email, target = await _seed_deferred(seed_session)
            email_id = email.id
            target_id = target.id

        _patch_current_parse(monkeypatch, _parsed_txn())
        async with maker() as preview_session:
            preview = await service.resolve_email_duplicate(
                preview_session,
                email_id,
                DuplicateResolutionRequest(transaction_id=target_id),
            )

        parse_started = asyncio.Event()
        release_parse = asyncio.Event()

        async def failing_parse(**_kwargs):
            parse_started.set()
            await release_parse.wait()
            return "stale parser detail", None, None, None

        monkeypatch.setattr(
            emails_web,
            "load_or_fetch_raw_email",
            AsyncMock(return_value=RawEmailLoadResult(RAW_EMAIL, None)),
        )
        monkeypatch.setattr(emails_web, "parse_email_by_kind", failing_parse)
        monkeypatch.setattr(emails_web, "_save_failed_email", lambda *_args: None)

        async def run_reparse():
            async with maker() as reparse_session:
                return await emails_web.reparse_email(
                    email_id, force_new=False, session=reparse_session
                )

        reparse_task = asyncio.create_task(run_reparse())
        await asyncio.wait_for(parse_started.wait(), timeout=2)

        async with maker() as apply_session:
            applied = await service.resolve_email_duplicate(
                apply_session,
                email_id,
                DuplicateResolutionRequest(
                    transaction_id=target_id,
                    apply=True,
                    preview_token=preview.preview_token,
                ),
            )
        assert applied.mode == "applied"

        release_parse.set()
        with pytest.raises(HTTPException) as parse_failure:
            await asyncio.wait_for(reparse_task, timeout=2)
        assert parse_failure.value.status_code == 422
        assert parse_failure.value.detail == "stale parser detail"

        async with maker() as check_session:
            stored_email = await check_session.get(Email, email_id)
            stored_target = await check_session.get(Transaction, target_id)
            assert stored_email is not None
            assert stored_email.status == "parsed"
            assert stored_email.error is None
            assert stored_target is not None
            assert stored_target.email_id == email_id
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_timed_email_accepts_untimed_target_on_date_touched_across_midnight(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    target.transaction_date = datetime.date(2030, 1, 1)
    target.transaction_time = None
    target.counterparty = "Synthetic Shop"
    target.reference_number = None
    await session.commit()
    _patch_current_parse(
        monkeypatch,
        _parsed_txn(
            transaction_date=datetime.date(2030, 1, 2),
            transaction_time=datetime.time(0, 5),
            reference_number=None,
            currency="INR",
        ),
    )

    preview = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert preview.status_code == 200, preview.text
    assert preview.json()["before"]["transaction_date"] == "2030-01-01"


@pytest.mark.anyio
async def test_no_reference_accepts_formatting_only_currency_difference(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    target.reference_number = None
    await session.commit()
    _patch_current_parse(
        monkeypatch,
        _parsed_txn(reference_number=None, currency="  inr  "),
    )

    preview = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert preview.status_code == 200, preview.text
    assert preview.json()["before"]["reference_number"] is None
    assert preview.json()["after"]["email_id"] == email.id


@pytest.mark.anyio
async def test_full_email_reference_enriches_short_sms_reference(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    # Counterparty deliberately disagrees: qualification comes from exact event
    # identity plus the shortened/full reference relationship, not ref equality.
    target.transaction_time = None
    target.counterparty = "Earlier SMS label"
    await session.commit()
    email_id, target_id = email.id, target.id
    _patch_current_parse(
        monkeypatch,
        _parsed_txn(
            transaction_time=None,
            counterparty="Current email label",
            reference_number="BANKPREFIX00000123",
        ),
    )

    preview = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={"transaction_id": target_id},
    )
    assert preview.status_code == 200, preview.text
    applied = await client.post(
        f"/api/emails/{email_id}/resolve-duplicate",
        json={
            "transaction_id": target_id,
            "apply": True,
            "preview_token": preview.json()["preview_token"],
        },
    )

    assert applied.status_code == 200, applied.text
    session.expire_all()
    stored = await session.get(Transaction, target_id)
    assert stored.reference_number == "BANKPREFIX00000123"
    assert await session.scalar(select(func.count()).select_from(Transaction)) == 1


@pytest.mark.anyio
async def test_full_email_reference_accepts_short_prefix_reference(
    client, session, monkeypatch
):
    email, target = await _seed_deferred(session)
    # Deliberately fail the date-only fuzzy counterparty gate so this exercises
    # only the conservative explicit shortened-reference fallback.
    target.transaction_time = None
    target.counterparty = "Earlier synthetic label"
    target.reference_number = "SYNTHPREFIX"
    await session.commit()
    _patch_current_parse(
        monkeypatch,
        _parsed_txn(
            transaction_time=None,
            counterparty="Current synthetic label",
            reference_number="SYNTHPREFIX987654",
        ),
    )

    preview = await client.post(
        f"/api/emails/{email.id}/resolve-duplicate",
        json={"transaction_id": target.id},
    )

    assert preview.status_code == 200, preview.text
    assert preview.json()["after"]["reference_number"] == "SYNTHPREFIX987654"


@pytest.mark.anyio
async def test_concurrent_sqlite_applies_attach_email_only_once(tmp_path, monkeypatch):
    """A SQLite write lock must serialize applies aimed at different rows."""
    from financial_dashboard.services import duplicate_resolution as service

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'duplicate-resolution.sqlite'}",
        connect_args={"timeout": 5},
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with maker() as seed_session:
            email, first_target = await _seed_deferred(seed_session)
            second_target = Transaction(
                bank="testbank",
                email_type="testbank_sms_debit_alert",
                direction="debit",
                amount=Decimal("42.15"),
                currency="INR",
                transaction_date=datetime.date(2030, 1, 2),
                transaction_time=datetime.time(10, 5),
                reference_number="SYNTHPREFIX",
                source="sms",
            )
            seed_session.add(second_target)
            await seed_session.commit()
            email_id = email.id
            first_id = first_target.id
            second_id = second_target.id

        _patch_current_parse(
            monkeypatch,
            _parsed_txn(reference_number="SYNTHPREFIX0123"),
        )

        async def preview(target_id: int) -> str:
            async with maker() as preview_session:
                response = await service.resolve_email_duplicate(
                    preview_session,
                    email_id,
                    DuplicateResolutionRequest(transaction_id=target_id),
                )
                return response.preview_token

        first_token, second_token = await asyncio.gather(
            preview(first_id), preview(second_id)
        )

        first_inside_mutation = asyncio.Event()
        release_first = asyncio.Event()
        original_apply = service.apply_transaction_enrichment

        async def blocking_apply(*args, **kwargs):
            if not first_inside_mutation.is_set():
                first_inside_mutation.set()
                await release_first.wait()
            return await original_apply(*args, **kwargs)

        monkeypatch.setattr(service, "apply_transaction_enrichment", blocking_apply)

        async def apply(target_id: int, token: str):
            async with maker() as apply_session:
                return await service.resolve_email_duplicate(
                    apply_session,
                    email_id,
                    DuplicateResolutionRequest(
                        transaction_id=target_id,
                        apply=True,
                        preview_token=token,
                    ),
                )

        first_apply = asyncio.create_task(apply(first_id, first_token))
        await asyncio.wait_for(first_inside_mutation.wait(), timeout=2)
        second_apply = asyncio.create_task(apply(second_id, second_token))
        await asyncio.sleep(0.1)
        assert not second_apply.done()

        release_first.set()
        first_result = await asyncio.wait_for(first_apply, timeout=2)
        assert first_result.mode == "applied"
        with pytest.raises(service.DuplicateResolutionError) as conflict:
            await asyncio.wait_for(second_apply, timeout=2)
        assert conflict.value.status_code == 409

        async with maker() as check_session:
            attached_ids = list(
                await check_session.scalars(
                    select(Transaction.id).where(Transaction.email_id == email_id)
                )
            )
            assert attached_ids == [first_id]
            assert (await check_session.get(Email, email_id)).status == "parsed"
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_concurrent_different_emails_cannot_claim_same_target(
    tmp_path, monkeypatch
):
    """A destination claim is exclusive even when source-row locks differ."""
    from financial_dashboard.services import duplicate_resolution as service

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'different-email-race.sqlite'}",
        connect_args={"timeout": 5},
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with maker() as seed_session:
            first_email, target = await _seed_deferred(seed_session)
            second_email = Email(
                provider="gmail",
                message_id="synthetic-deferred-message-2",
                sender=first_email.sender,
                subject=first_email.subject,
                status="skipped",
                error=DUP_DEFER_NOTE,
                rule_id=first_email.rule_id,
            )
            seed_session.add(second_email)
            await seed_session.commit()
            first_email_id = first_email.id
            second_email_id = second_email.id
            target_id = target.id

        _patch_current_parse(monkeypatch, _parsed_txn())

        async def preview(email_id: int) -> str:
            async with maker() as preview_session:
                response = await service.resolve_email_duplicate(
                    preview_session,
                    email_id,
                    DuplicateResolutionRequest(transaction_id=target_id),
                )
                return response.preview_token

        first_token, second_token = await asyncio.gather(
            preview(first_email_id), preview(second_email_id)
        )

        first_inside_mutation = asyncio.Event()
        release_first = asyncio.Event()
        original_apply = service.apply_transaction_enrichment

        async def blocking_apply(*args, **kwargs):
            if not first_inside_mutation.is_set():
                first_inside_mutation.set()
                await release_first.wait()
            return await original_apply(*args, **kwargs)

        monkeypatch.setattr(service, "apply_transaction_enrichment", blocking_apply)

        async def apply(email_id: int, token: str):
            async with maker() as apply_session:
                return await service.resolve_email_duplicate(
                    apply_session,
                    email_id,
                    DuplicateResolutionRequest(
                        transaction_id=target_id,
                        apply=True,
                        preview_token=token,
                    ),
                )

        first_apply = asyncio.create_task(apply(first_email_id, first_token))
        await asyncio.wait_for(first_inside_mutation.wait(), timeout=2)
        second_apply = asyncio.create_task(apply(second_email_id, second_token))
        await asyncio.sleep(0.1)
        assert not second_apply.done()

        release_first.set()
        first_result = await asyncio.wait_for(first_apply, timeout=2)
        assert first_result.mode == "applied"
        with pytest.raises(service.DuplicateResolutionError) as conflict:
            await asyncio.wait_for(second_apply, timeout=2)
        assert conflict.value.status_code == 409

        async with maker() as check_session:
            stored_target = await check_session.get(Transaction, target_id)
            stored_first = await check_session.get(Email, first_email_id)
            stored_second = await check_session.get(Email, second_email_id)
            assert stored_target is not None
            assert stored_target.email_id == first_email_id
            assert stored_first is not None and stored_first.status == "parsed"
            assert stored_second is not None and stored_second.status == "skipped"
            assert stored_second.error == DUP_DEFER_NOTE
    finally:
        await engine.dispose()
