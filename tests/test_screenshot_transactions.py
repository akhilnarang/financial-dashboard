"""UPI screenshots become provisional rows. All names, amounts and UTRs are synthetic."""

import datetime
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from bank_statement_parser.models import BankTransaction, ParsedBankStatement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.db as db_package
from financial_dashboard.db.models import (
    Account,
    AuditInteraction,
    Setting,
    TelegramConversation,
    TelegramOutboundDelivery,
    Transaction,
)
from financial_dashboard.services import screenshot_transactions, telegram
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services import transaction_attachments
from financial_dashboard.services.assistant.contracts import ScreenshotTransaction
from financial_dashboard.services.assistant.message_context import resolve_reply
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.assistant.orchestrator import (
    _process_screenshot_interaction,
    _process_text_interaction,
    run_pending_confirmation,
)
from financial_dashboard.services.categorization import openai_provider
from financial_dashboard.services.statements import bank as bank_statements
from financial_dashboard.services.txn_merge import merge_transaction
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio

CHAT = 77
UTR = "123456789012"
PNG = b"\x89PNG\r\n\x1a\nsynthetic"
REAL_DISPATCH = telegram.dispatch_saved_delivery


class Shot(NamedTuple):
    maker: async_sessionmaker[AsyncSession]
    root: Path
    vision_calls: list[dict[str, str | bytes]]


class Asked(NamedTuple):
    conversation_id: int
    text: str
    buttons: list[str]


def _read(**fields: str | Decimal | datetime.datetime | None) -> ScreenshotTransaction:
    base: dict[str, str | Decimal | datetime.datetime | None] = {
        "status": "success",
        "direction": "debit",
        "amount": Decimal("250.00"),
        "counterparty": "Test Merchant",
        "vpa": "merchant@testbank",
        "utr": UTR,
        "occurred_at": datetime.datetime(2026, 10, 20, 14, 5),
        "app": "gpay",
        "account_hint": "SBI 4321",
    }
    return ScreenshotTransaction.model_validate(base | fields)


@pytest.fixture
async def shot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[Shot]:
    """Two SBI accounts, Telegram on, and a vision call that records its input."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    for module in (db_package, telegram, screenshot_transactions, bank_statements):
        monkeypatch.setattr(module, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.config.settings.transaction_attachment_root",
        str(tmp_path),
    )
    settings_service._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "token",
            "telegram.chat_id": str(CHAT),
            "telegram.assistant_enabled": "true",
        }
    )
    async with maker() as s:
        s.add_all(
            [
                Setting(key="telegram.chat_id", value=str(CHAT)),
                Account(
                    id=1,
                    bank="sbi",
                    label="SBI savings",
                    type="bank_account",
                    account_number="XXXXXX4321",
                ),
                Account(
                    id=2,
                    bank="sbi",
                    label="SBI other",
                    type="bank_account",
                    account_number="XXXXXX9999",
                ),
            ]
        )
        await s.commit()

    async def no_dispatch(delivery_id: int) -> bool:
        return True

    monkeypatch.setattr(telegram, "dispatch_saved_delivery", no_dispatch)
    yield Shot(maker, tmp_path, [])
    await engine.dispose()
    holder.close()


async def _send(
    shot: Shot, monkeypatch: pytest.MonkeyPatch, read: ScreenshotTransaction
) -> Asked:
    """Send one screenshot through the bot and return the question it asks."""

    async def fake_download(
        url: str, *, transaction_id: int, declared_size: int | None
    ) -> transaction_attachments.StoredAttachment:
        path = shot.root / "shot.png"
        path.write_bytes(PNG)
        return transaction_attachments.StoredAttachment("shot.png", path, "image/png")

    async def fake_vision(
        *, image: bytes, media_type: str, api_key: str, model: str, base_url: str
    ) -> ScreenshotTransaction:
        shot.vision_calls.append({"image": image, "model": model})
        return read

    async def get_file(file_id: str) -> SimpleNamespace:
        return SimpleNamespace(file_path="https://api.telegram.org/file/botT/p")

    monkeypatch.setattr(transaction_attachments, "download_attachment", fake_download)
    monkeypatch.setattr(openai_provider, "read_screenshot", fake_vision)
    async with shot.maker() as s:
        conversation = TelegramConversation(
            chat_id=CHAT,
            started_by="screenshot",
            status="active",
            expires_at=datetime.datetime.now(datetime.UTC)
            + datetime.timedelta(hours=1),
        )
        s.add(conversation)
        await s.flush()
        interaction = AuditInteraction(
            inbound_chat_id=CHAT,
            trigger="screenshot",
            status="processing",
            worker_token="worker",
            conversation_id=conversation.id,
        )
        s.add(interaction)
        await s.commit()
    await _process_screenshot_interaction(
        bot=SimpleNamespace(get_file=get_file),
        interaction_id=interaction.id,
        worker_token="worker",
        conversation_id=conversation.id,
        recipient_chat_id=CHAT,
        file_id="file-id",
        declared_size=len(PNG),
    )
    return await _asked(shot, conversation.id, interaction.id)


async def _asked(shot: Shot, conversation_id: int, interaction_id: int) -> Asked:
    async with shot.maker() as s:
        delivery = await s.scalar(
            select(TelegramOutboundDelivery).where(
                TelegramOutboundDelivery.interaction_id == interaction_id
            )
        )
    assert delivery is not None
    rows = json.loads(delivery.reply_markup_json or "[]")
    return Asked(
        conversation_id, delivery.text, [b["text"] for row in rows for b in row]
    )


async def _yes(shot: Shot, conversation_id: int) -> str:
    async with shot.maker() as s:
        conversation = await s.get(TelegramConversation, conversation_id)
        assert conversation is not None
        result = await run_pending_confirmation(
            s,
            conversation_id=conversation_id,
            state_hash=conversation.pending_confirmation_state_hash or "",
            user_message="yes",
            replied_to_interaction_id=(
                conversation.pending_confirmation_source_interaction_id or 0
            ),
            authorized_chat_id=CHAT,
        )
        await s.commit()
    assert isinstance(result, str)
    return result


async def _reply(shot: Shot, conversation_id: int, text: str) -> Asked:
    """Reply to the pending screenshot question and return the next question."""
    async with shot.maker() as s:
        conversation = await s.get(TelegramConversation, conversation_id)
        assert conversation is not None
        source = conversation.pending_confirmation_source_interaction_id
        reply = AuditInteraction(
            inbound_chat_id=CHAT,
            trigger="reply",
            status="processing",
            worker_token="reply-worker",
            conversation_id=conversation_id,
        )
        s.add(reply)
        await s.commit()
    await _process_text_interaction(
        interaction_id=reply.id,
        worker_token="reply-worker",
        conversation_id=conversation_id,
        transaction_id=None,
        user_text=text,
        replied_to_interaction_id=source,
        recipient_chat_id=CHAT,
    )
    return await _asked(shot, conversation_id, reply.id)


async def _rows(shot: Shot) -> list[Transaction]:
    async with shot.maker() as s:
        return list(
            (await s.scalars(select(Transaction).order_by(Transaction.id))).all()
        )


async def test_failed_screenshot_is_refused(
    shot: Shot, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = await _send(shot, monkeypatch, _read(status="failed"))

    assert shot.vision_calls == [{"image": PNG, "model": "gpt-6.1-luna"}]
    assert "failed payment" in asked.text
    assert asked.buttons == []
    assert not (shot.root / "shot.png").exists()
    assert await _rows(shot) == []


@pytest.mark.parametrize("amount", [Decimal("0.001"), Decimal("1e30")])
async def test_missing_fields_ask_and_a_reply_fills_them_before_yes_saves(
    shot: Shot, monkeypatch: pytest.MonkeyPatch, amount: Decimal
) -> None:
    asked = await _send(
        shot,
        monkeypatch,
        _read(amount=amount, utr="12345678901", account_hint="SBI 4321XXXX"),
    )

    # An amount below one paisa, or too large to store, is not valid.
    assert "amount, UTR (12 digits), account (last 4 digits)" in asked.text
    assert asked.buttons == ["Skip"]
    with pytest.raises(MutationRejected):
        await _yes(shot, asked.conversation_id)

    # A name over the length limit, a negative amount and an amount below one
    # paisa are dropped. A comma inside an amount groups its digits.
    proposed = await _reply(
        shot,
        asked.conversation_id,
        f"utr {UTR}, account 4321, amount is Rs. 1,250.00., amount -950, it was ₹950.001, "
        "it was ₹1,250.001, "
        f"name {'x' * 250}",
    )
    assert "Save this as a provisional transaction?" in proposed.text
    assert proposed.buttons == ["Yes", "Skip"]

    assert (await _yes(shot, asked.conversation_id)).startswith("Saved #")
    [row] = await _rows(shot)
    assert (
        row.account_id,
        row.account_mask,
        row.bank,
        row.source,
        row.email_type,
        row.channel,
        row.direction,
        row.amount,
        row.reference_number,
        row.counterparty,
        row.raw_description,
        row.transaction_date,
        row.transaction_time,
        row.attachment_path,
    ) == (
        1,
        "XX4321",
        "sbi",
        "screenshot",
        "upi_screenshot",
        "upi",
        "debit",
        Decimal("1250.00"),
        UTR,
        "Test Merchant",
        "merchant@testbank",
        datetime.date(2026, 10, 20),
        datetime.time(14, 5),
        "shot.png",
    )
    assert (shot.root / "shot.png").read_bytes() == PNG


@pytest.mark.parametrize(
    "case",
    [
        "same_utr",
        "voided_utr",
        "voided_since",
        "other_utr",
        "credit_leg",
        "maybe_new",
        "maybe_moved",
        "maybe_no_utr",
    ],
)
async def test_duplicate_utr_offers_to_attach_instead(
    shot: Shot, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    async with shot.maker() as s:
        # The same account, amount and day, but another UTR: another payment.
        other = Transaction(
            bank="sbi",
            account_id=1,
            email_type="sbi_upi_alert",
            direction="debit",
            amount=Decimal("250.00"),
            reference_number="210987654321",
            transaction_date=datetime.date(2026, 10, 20),
            source="sms",
        )
        # The credit leg of a self-transfer carries the same UTR.
        existing = Transaction(
            bank="sbi",
            email_type="sbi_upi_alert",
            direction="credit" if case == "credit_leg" else "debit",
            amount=Decimal("250.00"),
            reference_number=UTR,
            source="sms",
            voided=case == "voided_utr",
        )
        # A failed attempt the same day is voided; the guess skips it.
        failed = Transaction(
            bank="sbi",
            account_id=1,
            email_type="sbi_upi_alert",
            direction="debit",
            amount=Decimal("250.00"),
            transaction_date=datetime.date(2026, 10, 20),
            source="sms",
            voided=True,
        )
        # The same account, amount and day, and no UTR: maybe the same payment.
        maybe = Transaction(
            bank="sbi",
            account_id=1,
            email_type="sbi_upi_alert",
            direction="debit",
            amount=Decimal("250.00"),
            counterparty="TEST MERCHANT",
            transaction_date=datetime.date(2026, 10, 20),
            source="sms",
        )
        s.add_all(
            {
                "other_utr": [other],
                "maybe_new": [other, failed, maybe],
                "maybe_moved": [other, failed, maybe],
                "maybe_no_utr": [other, failed, maybe],
            }.get(case, [other, existing])
        )
        await s.commit()

    asked = await _send(
        shot, monkeypatch, _read(utr=None) if case == "maybe_no_utr" else _read()
    )

    if case.startswith("maybe"):
        assert f"may already be recorded as #{maybe.id}" in asked.text
        assert asked.buttons == [f"Attach to #{maybe.id}", "Skip"]
    if case == "maybe_no_utr":
        return
    if case == "maybe_moved":
        # The row changes before the Yes, so it no longer matches.
        async with shot.maker() as s:
            row = await s.get(Transaction, maybe.id)
            assert row is not None
            row.amount = Decimal("260.00")
            await s.commit()
        with pytest.raises(MutationRejected, match="no longer matches"):
            await _yes(shot, asked.conversation_id)
        return
    if case == "maybe_new":
        asked = await _reply(shot, asked.conversation_id, "new")
        assert asked.buttons == ["Yes", "Skip"]
        assert (await _yes(shot, asked.conversation_id)).startswith("Saved #")
        assert len(await _rows(shot)) == 4
        return
    if case == "voided_since":
        # The row is voided after the question, so the old tap must not un-void it.
        async with shot.maker() as s:
            row = await s.get(Transaction, existing.id)
            assert row is not None
            row.voided = True
            await s.commit()
        with pytest.raises(MutationRejected, match="no longer matches"):
            await _yes(shot, asked.conversation_id)
        return
    if case not in ("same_utr", "voided_utr"):
        assert "Save this as a provisional transaction?" in asked.text
        return

    # A voided row with the screenshot's UTR is the same payment, which the
    # success screenshot shows did go through.
    voided = case == "voided_utr"
    assert (
        f"recorded as #{existing.id}, which is voided"
        if voided
        else f"already recorded as #{existing.id}"
    ) in asked.text
    assert asked.buttons == [f"Attach to #{existing.id}", "Skip"]
    assert await _yes(shot, asked.conversation_id) == (
        f"Attached the screenshot to #{existing.id}"
        + (" and un-voided it." if voided else ".")
    )
    stored = {row.id: (row.attachment_path, row.voided) for row in await _rows(shot)}
    assert stored == {other.id: (None, False), existing.id: ("shot.png", False)}


def _screenshot_row() -> Transaction:
    return Transaction(
        bank="sbi",
        account_id=1,
        source="screenshot",
        email_type="upi_screenshot",
        channel="upi",
        direction="debit",
        amount=Decimal("250.00"),
        currency="INR",
        reference_number=UTR,
        counterparty="Test Merchant",
        transaction_date=datetime.date(2026, 10, 20),
        transaction_time=datetime.time(14, 5),
    )


async def test_alert_with_the_same_utr_enriches_the_screenshot_row(
    shot: Shot,
) -> None:
    async with shot.maker() as s:
        s.add(_screenshot_row())
        await s.flush()
        # An alert a minute later with another UTR is another payment.
        other, _, _ = await merge_transaction(
            s,
            "sms",
            {
                "bank": "sbi",
                "email_type": "sbi_upi_alert",
                "direction": "debit",
                "amount": Decimal("250.00"),
                "currency": "INR",
                "reference_number": "210987654321",
                "transaction_date": datetime.date(2026, 10, 20),
                "transaction_time": datetime.time(14, 6),
                "counterparty": None,
                "balance": Decimal("750.00"),
                "channel": "upi",
            },
        )
        outcome, row, _ = await merge_transaction(
            s,
            "sms",
            {
                "bank": "sbi",
                "email_type": "sbi_upi_alert",
                "direction": "debit",
                "amount": Decimal("250.00"),
                "currency": "INR",
                "reference_number": UTR,
                "transaction_date": datetime.date(2026, 10, 20),
                "transaction_time": datetime.time(14, 6),
                "counterparty": None,
                "balance": Decimal("1000.00"),
                "channel": "upi",
            },
        )
        # The email copy of the same alert keeps the screenshot source.
        await merge_transaction(
            s,
            "email",
            {
                "bank": "sbi",
                "email_type": "sbi_upi_alert",
                "direction": "debit",
                "amount": Decimal("250.00"),
                "currency": "INR",
                "reference_number": UTR,
                "transaction_date": datetime.date(2026, 10, 20),
                "transaction_time": datetime.time(14, 6),
                "counterparty": None,
                "channel": "upi",
            },
        )
        await s.commit()

    assert (other, outcome) == ("created", "enriched")
    stored, _ = await _rows(shot)
    assert (stored.id, stored.source, stored.balance) == (
        row.id if row else None,
        "screenshot+sms+email",
        Decimal("1000.00"),
    )


@pytest.mark.parametrize("utr", [UTR, "210987654321"])
async def test_statement_line_with_the_utr_confirms_the_screenshot_row(
    shot: Shot, utr: str
) -> None:
    async with shot.maker() as s:
        s.add(_screenshot_row())
        await s.commit()
    # A line the same day to the same payee with another UTR is another payment.
    statement = ParsedBankStatement(
        file="synthetic.pdf",
        bank="sbi",
        transactions=[
            BankTransaction(
                date="21/10/2026" if utr == UTR else "20/10/2026",
                narration=f"UPI/DR/{utr}/TEST MERCHANT",
                amount="250.00",
                transaction_type="debit",
                reference_number=utr,
                channel="upi",
            )
        ],
    )

    recon = bank_statements.reconcile_bank_statement(statement, await _rows(shot), 1)
    await bank_statements.enrich_matched_transactions(recon)
    if utr != UTR:
        assert recon["matched"] == []
        assert [row.source for row in await _rows(shot)] == ["screenshot"]
        return

    assert [entry["db_txn_id"] for entry in recon["matched"]] == [1]
    assert recon["missing"] == []
    [stored] = await _rows(shot)
    assert stored.source == "screenshot+bank_statement"


async def test_flag_sweep_asks_once_and_a_reply_targets_the_row(
    shot: Shot, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []

    async def fake_send(
        app: object, *, chat_id: int, text: str, **_: object
    ) -> SimpleNamespace:
        sent.append(text)
        return SimpleNamespace(message_id=500 + len(sent))

    monkeypatch.setattr(telegram, "tg_app", object())
    monkeypatch.setattr(telegram, "_send_with_retry", fake_send)
    monkeypatch.setattr(telegram, "dispatch_saved_delivery", REAL_DISPATCH)
    today = datetime.date.today()
    async with shot.maker() as s:
        old = _screenshot_row()
        old.transaction_date = today - datetime.timedelta(days=61)
        recent = _screenshot_row()
        recent.reference_number = "210987654321"
        recent.transaction_date = today - datetime.timedelta(days=30)
        s.add_all([old, recent])
        await s.commit()

    assert await screenshot_transactions.run_screenshot_flag_sweep() == 1
    assert await screenshot_transactions.run_screenshot_flag_sweep() == 0
    assert len(sent) == 1 and sent[0].startswith(f"#{old.id} ₹250.00 to Test Merchant")

    assert "What happened? Reply to this message.\n\nRef: " in sent[0]
    async with shot.maker() as s:
        delivery = await s.scalar(select(TelegramOutboundDelivery))
        context = await resolve_reply(s, chat_id=CHAT, message_id=501)
    assert delivery is not None and delivery.reply_markup_json is None
    assert context is not None and context.transaction_id == old.id
