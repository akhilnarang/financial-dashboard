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
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.db as db_package
from financial_dashboard.db.models import (
    Account,
    AuditInteraction,
    Card,
    Setting,
    TelegramConversation,
    TelegramOutboundDelivery,
    Transaction,
)
from financial_dashboard.services import screenshot_transactions, telegram
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services import transaction_attachments
from financial_dashboard.services.assistant.contracts import (
    ScreenshotTransaction,
    UpiParty,
)
from financial_dashboard.services.assistant.message_context import resolve_reply
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.assistant.orchestrator import (
    _process_screenshot_interaction,
    _process_text_interaction,
    run_pending_confirmation,
)
from financial_dashboard.services.categorization import openai_provider
from financial_dashboard.services.categorization.engine import categorize_one
from financial_dashboard.services.statements import bank as bank_statements
from financial_dashboard.services.statements import cc as cc_statements
from financial_dashboard.services.txn_merge import find_match, merge_transaction
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
    for module in (
        db_package,
        telegram,
        screenshot_transactions,
        bank_statements,
        cc_statements,
    ):
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
    shot: Shot,
    monkeypatch: pytest.MonkeyPatch,
    read: ScreenshotTransaction,
    caption: str | None = None,
) -> Asked:
    """Send one screenshot through the bot and return the question it asks."""

    async def fake_download(
        url: str, *, transaction_id: int, declared_size: int | None
    ) -> transaction_attachments.StoredAttachment:
        path = shot.root / "shot.png"
        path.write_bytes(PNG)
        return transaction_attachments.StoredAttachment("shot.png", path, "image/png")

    async def fake_vision(
        *,
        image: bytes,
        media_type: str,
        api_key: str,
        model: str,
        base_url: str,
        note: str | None = None,
        current: ScreenshotTransaction | None = None,
    ) -> ScreenshotTransaction:
        shot.vision_calls.append({"image": image, "model": model, "note": note})
        return current or read

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
        caption=caption,
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

    assert shot.vision_calls == [{"image": PNG, "model": "gpt-6-luna", "note": None}]
    assert "failed payment" in asked.text
    assert "saved nothing" in asked.text
    assert asked.buttons == []
    assert not (shot.root / "shot.png").exists()
    assert await _rows(shot) == []

    pending = await _send(shot, monkeypatch, _read(status="pending"))
    assert "pending payment" in pending.text
    assert "saved nothing" in pending.text
    assert pending.buttons == []
    assert await _rows(shot) == []

    async def raise_empty_value_error(**kwargs: object) -> ScreenshotTransaction:
        raise ValueError(
            "https://api.telegram.org/file/botSECRET_BOT_TOKEN/photos/1.jpg"
        )

    monkeypatch.setattr(openai_provider, "read_screenshot", raise_empty_value_error)
    async with shot.maker() as s:
        interaction = AuditInteraction(
            inbound_chat_id=CHAT,
            trigger="screenshot",
            status="processing",
            worker_token="worker-err",
            conversation_id=asked.conversation_id,
        )
        s.add(interaction)
        await s.commit()

    async def get_file(file_id: str) -> SimpleNamespace:
        return SimpleNamespace(file_path="https://api.telegram.org/file/botT/p")

    await _process_screenshot_interaction(
        bot=SimpleNamespace(get_file=get_file),
        interaction_id=interaction.id,
        worker_token="worker-err",
        conversation_id=asked.conversation_id,
        recipient_chat_id=CHAT,
        file_id="file-id",
        declared_size=len(PNG),
        caption=None,
    )
    async with shot.maker() as s:
        stored_interaction = await s.get(AuditInteraction, interaction.id)
    assert stored_interaction is not None
    assert stored_interaction.error_code == "screenshot_unreadable"
    assert stored_interaction.model == "gpt-6-luna"
    assert stored_interaction.error_detail is not None
    assert "Traceback" in stored_interaction.error_detail
    assert "ValueError" in stored_interaction.error_detail
    assert "bot[REDACTED]" in stored_interaction.error_detail
    assert "SECRET_BOT_TOKEN" not in stored_interaction.error_detail


def _card_read(**fields: str | Decimal | None) -> ScreenshotTransaction:
    """Build one card notification read, with synthetic values."""
    base: dict[str, str | Decimal | None] = {
        "status": "reversed",
        "direction": "debit",
        "channel": "card",
        "app": "other",
        "amount": Decimal("100.00"),
        "counterparty": "Example Merchant",
        "vpa": None,
        "utr": "",
        "account_hint": "OneCard · Groceries",
    }
    return _read(**(base | fields))


async def test_card_reversal_screenshot_saves_refund_and_merges_with_alert(
    shot: Shot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a card refund screenshot saves a refund row and merges its alert.

    Without this path, a card refund has no row. The row must name the card
    account, carry the refund category, and enrich with the later alert. The
    same refund on the card statement must confirm the row, not duplicate it.
    """
    async with shot.maker() as s:
        s.add(Account(id=3, bank="OneCard", label="OneCard CC", type="credit_card"))
        s.add(Card(id=1, account_id=3, card_mask="XX9999", is_primary=True))
        await s.commit()

    # A card spend or bill payment has its own alert, so only a refund saves.
    spend = await _send(shot, monkeypatch, _card_read(status="success"))
    assert "not a refund" in spend.text
    assert spend.buttons == []

    asked = await _send(shot, monkeypatch, _card_read())

    assert asked.text.startswith("Card refund: ₹100.00 from")
    assert "UTR" not in asked.text
    assert asked.buttons == ["Yes", "Skip"]
    assert (await _yes(shot, asked.conversation_id)).startswith("Saved #")
    [row] = await _rows(shot)
    assert (
        row.account_id,
        row.account_mask,
        row.card_id,
        row.card_mask,
        row.bank,
        row.direction,
        row.channel,
        row.email_type,
        row.category,
        row.category_method,
        row.category_confidence,
        row.source,
        row.reference_number,
    ) == (
        3,
        None,
        1,
        "XX9999",
        "onecard",
        "credit",
        "card",
        "card_refund_screenshot",
        None,
        None,
        None,
        "screenshot",
        None,
    )

    # The rule pass names a refund screenshot before any model call.
    async with shot.maker() as s:
        txn = await s.get(Transaction, row.id)
        assert await categorize_one(s, txn, use_llm=False) == "rule"
        await s.commit()
    [row] = await _rows(shot)
    assert (row.category, row.category_method) == ("refund", "rule")

    # The same screenshot again offers the stored row, not a second row.
    again = await _send(shot, monkeypatch, _card_read())
    assert f"may already be recorded as #{row.id}" in again.text
    assert len(await _rows(shot)) == 1

    # The alert for the same refund enriches the screenshot row.
    async with shot.maker() as s:
        outcome, matched, _ = await merge_transaction(s, "sms", _alert("100.00"))
        # Another amount is another refund, so it becomes a new row.
        other, _, _ = await merge_transaction(s, "email", _alert("200.00"))
        repeat, _, _ = await merge_transaction(s, "sms", _alert("100.00"))
        await s.commit()
    assert (outcome, matched.id if matched else None) == ("enriched", row.id)
    assert matched is not None and matched.source == "screenshot+sms"
    assert other == "created"
    assert repeat == "enriched"
    assert len(await _rows(shot)) == 2

    statement = SimpleNamespace(
        bank="onecard",
        transactions=[],
        payments_refunds=[
            CcTransaction(
                date="20/10/2026",
                narration="EXAMPLE MERCHANT REFUND",
                amount="100.00",
                card_number="XX9999",
                person=None,
                transaction_type="credit",
            )
        ],
        payments_refunds_total="100.00",
        card_summaries=[],
        possible_adjustment_pairs=[],
        overall_total="0.00",
        overall_reward_points="0",
    )
    async with shot.maker() as s:
        masks = await cc_statements.load_account_card_masks(s, 3)
    recon = cc_statements.reconcile_statement(statement, await _rows(shot), 3, masks)
    # A line that could be one of several rows confirms none of them.
    [entry] = recon["matched"]
    await cc_statements.enrich_matched_transactions(
        {"matched": [entry | {"candidate_count": 2}]}
    )
    assert (await _rows(shot))[0].source == "screenshot+sms"
    await cc_statements.enrich_matched_transactions(recon)
    assert [entry["db_txn_id"] for entry in recon["matched"]] == [row.id]
    assert recon["missing"] == []
    stored, _ = await _rows(shot)
    assert stored.source == "screenshot+sms+cc_statement"

    # A bank account credit of the same amount is not the card refund.
    async with shot.maker() as s:
        savings = await find_match(
            s, _alert("100.00") | {"account_mask": "XX1234"}, "sms"
        )
    assert savings.action == "insert"

    # A row with no stored card takes only an alert that names no card.
    async with shot.maker() as s:
        bare = await s.get(Transaction, row.id)
        assert bare is not None
        bare.card_mask = None
        await s.flush()
        debit_card = await find_match(
            s, _alert("100.00") | {"card_mask": "XX5555"}, "sms"
        )
        await s.rollback()
    assert debit_card.action == "insert"

    # An alert on another card is another refund, even on the same account.
    async with shot.maker() as s:
        wrong_card, _, _ = await merge_transaction(
            s, "sms", _alert("100.00") | {"card_mask": "XX1111"}
        )
        await s.commit()
    assert wrong_card == "created"


async def test_card_hint_without_digits_asks_for_the_card_when_ambiguous(
    shot: Shot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an ambiguous card hint asks for the card and saves nothing.

    Without the guard, a refund can land on the wrong card account.
    """
    async with shot.maker() as s:
        s.add_all(
            [
                Account(id=3, bank="onecard", label="OneCard A", type="credit_card"),
                Account(id=4, bank="onecard", label="OneCard B", type="credit_card"),
                Card(id=1, account_id=3, card_mask="XX9999", is_primary=True),
                Card(id=2, account_id=4, card_mask="XX8888", is_primary=True),
            ]
        )
        await s.commit()

    asked = await _send(shot, monkeypatch, _card_read(account_hint="OneCard"))

    assert "Missing or not valid: card" in asked.text
    assert "Which card: OneCard A 9999, OneCard B 8888?" in asked.text
    assert 'for example "card 1234, amount 100"' in asked.text
    assert asked.buttons == ["Skip"]
    with pytest.raises(MutationRejected):
        await _yes(shot, asked.conversation_id)
    assert await _rows(shot) == []

    # Replying with the card last 4 resolves the card and lets Yes save.
    proposed = await _reply(shot, asked.conversation_id, "card 9999")
    assert "Save this as a provisional transaction?" in proposed.text
    assert (await _yes(shot, asked.conversation_id)).startswith("Saved #")
    [saved] = await _rows(shot)
    assert (saved.account_id, saved.card_id) == (3, 1)


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
    assert "amount, UTR (12 digits), account (last 4 digits or bank)" in asked.text
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


async def test_payee_receipt_saves_a_credit_on_the_named_bank(
    shot: Shot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a receipt paid to the owner saves a credit on the named bank.

    Without the From and To blocks, a receipt paid to the owner reads as a
    payment the owner made.
    """
    settings_service._cache["categorization.self_identifiers"] = (
        "JANE DOE, jane@testyes"
    )
    paid_at = (
        datetime.datetime.now(screenshot_transactions.IST) - datetime.timedelta(days=1)
    ).replace(hour=18, minute=16, second=0, microsecond=0, tzinfo=None)
    receipt = _read(
        direction="debit",
        amount=Decimal("1234.00"),
        counterparty="JANE DOE",
        vpa="jane@testyes",
        utr=None,
        occurred_at=paid_at,
        app="supermoney",
        account_hint=None,
        payer=UpiParty(name="ALEX SMITH", vpa="test0000@upi", bank_text=None),
        payee=UpiParty(
            name="JANE DOE", vpa="jane@testyes", bank_text="State Bank of India"
        ),
    )

    # With two active SBI accounts, a bank name alone is ambiguous.
    ambiguous = await _send(shot, monkeypatch, receipt)
    assert "received ₹1,234.00 from ALEX SMITH (test0000@upi)" in ambiguous.text
    assert "UTR (12 digits), account (last 4 digits or bank)" in ambiguous.text
    assert "Which account: SBI savings 4321, SBI other 9999?" in ambiguous.text

    # A single name can be a friend's name, so it never marks the owner.
    lone_name = screenshot_transactions.orient_read(receipt, ("DOE",))
    assert lone_name.direction == "debit"
    name_only = receipt.model_copy(update={"payee": UpiParty(name="JANE Q DOE")})
    two_words = screenshot_transactions.orient_read(name_only, ("JANE", "DOE"))
    assert two_words.direction == "credit"
    owner = ("JANE DOE", "jane@testyes")
    # A refund to the owner stays a credit, and shown last 4 digits stay.
    refund = receipt.model_copy(
        update={"status": "reversed", "payer": receipt.payee, "payee": receipt.payer}
    )
    assert screenshot_transactions.orient_read(refund, owner).direction == "credit"
    with_last4 = receipt.model_copy(update={"account_hint": "SBI 4321"})
    oriented = screenshot_transactions.orient_read(with_last4, owner)
    assert oriented.account_hint == "SBI 4321"
    sender_digits = with_last4.model_copy(
        update={"payer": UpiParty(name="ALEX SMITH", bank_text="HDFC 4321")}
    )
    assert screenshot_transactions.orient_read(sender_digits, owner).account_hint == (
        "State Bank of India"
    )
    # A sender's bank never names the owner's account, and a partial block
    # keeps the name already read.
    sender_bank = receipt.model_copy(
        update={
            "account_hint": "HDFC",
            "counterparty": "ALEX SMITH",
            "payer": UpiParty(bank_text="HDFC"),
            "payee": UpiParty(name="JANE DOE", vpa="jane@testyes"),
        }
    )
    oriented = screenshot_transactions.orient_read(sender_bank, owner)
    assert (oriented.account_hint, oriented.counterparty) == (None, "ALEX SMITH")
    # A free 12-digit number never replaces a valid UTR.
    kept = screenshot_transactions.apply_corrections(
        receipt.model_copy(update={"utr": UTR}), "my account number is 999999999999"
    )
    assert kept.utr == UTR
    typed = screenshot_transactions.apply_corrections(
        receipt, f"utr {UTR}; my account number is 999999999999"
    )
    assert typed.utr == UTR
    account_number = screenshot_transactions.apply_corrections(
        receipt, "my account number is 999999999999"
    )
    assert account_number.utr is None
    assert (
        screenshot_transactions.apply_corrections(receipt, f"UPI ref ID is {UTR}").utr
        == UTR
    )

    # With one active SBI account, "State Bank of India" resolves the account.
    async with shot.maker() as s:
        other = await s.get(Account, 2)
        assert other is not None
        other.active = False
        await s.commit()

    caption = "Sent you payment via UPI. Let me know when you get it"
    asked = await _send(shot, monkeypatch, receipt, caption)
    assert shot.vision_calls[-1]["note"] == caption
    assert (
        "UPI screenshot (supermoney): received ₹1,234.00 from ALEX SMITH (test0000@upi)"
        in asked.text
    )
    assert f"{paid_at:%Y-%m-%d %H:%M} · SBI savings" in asked.text
    assert "Missing or not valid: UTR (12 digits)." in asked.text

    async def vision_down(**kwargs: object) -> ScreenshotTransaction:
        raise ValueError("vision model down")

    # When the model fails, the reply still gets the strict parser.
    monkeypatch.setattr(openai_provider, "read_screenshot", vision_down)
    unchanged = await _reply(shot, asked.conversation_id, "thanks")
    assert unchanged.text.startswith("I found no new value in your reply.\n")

    seen: list[dict] = []

    async def vision_with_note(
        *,
        image: bytes,
        media_type: str,
        api_key: str,
        model: str,
        base_url: str,
        note: str | None = None,
        current: ScreenshotTransaction | None = None,
    ) -> ScreenshotTransaction:
        seen.append({"image": image, "note": note, "current": current})
        assert current is not None
        return current.model_copy(update={"utr": UTR})

    # A plain-language reply goes to the model with the image and current read.
    monkeypatch.setattr(openai_provider, "read_screenshot", vision_with_note)
    note = "The UPI ref ID is printed below. I have only one SBI account."
    proposed = await _reply(shot, asked.conversation_id, note)
    assert [(c["image"], c["note"], c["current"].utr) for c in seen] == [
        (PNG, note, None)
    ]
    assert "Save this as a provisional transaction?" in proposed.text
    assert (await _yes(shot, asked.conversation_id)).startswith("Saved #")
    [row] = await _rows(shot)
    assert (
        row.account_id,
        row.direction,
        row.amount,
        row.reference_number,
        row.counterparty,
        row.raw_description,
        row.transaction_date,
        row.transaction_time,
    ) == (
        1,
        "credit",
        Decimal("1234.00"),
        UTR,
        "ALEX SMITH",
        "test0000@upi",
        paid_at.date(),
        datetime.time(18, 16),
    )


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
        "corrupt",
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
    if case == "corrupt":
        # A reply to a question whose stored action no longer parses still
        # gets an answer, and the action is gone.
        async with shot.maker() as s:
            conversation = await s.get(TelegramConversation, asked.conversation_id)
            assert conversation is not None
            conversation.pending_confirmation_json = (
                '{"action":{"kind":"screenshot_transaction"}}'
            )
            await s.commit()
        answered = await _reply(shot, asked.conversation_id, "new")
        assert "no longer valid" in answered.text
        async with shot.maker() as s:
            conversation = await s.get(TelegramConversation, asked.conversation_id)
            assert conversation is not None
            assert conversation.pending_confirmation_json is None
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


def _alert(amount: str) -> dict:
    """Build one OneCard refund alert for the merge checks."""
    return {
        "bank": "onecard",
        "email_type": "onecard_cc_refund",
        "direction": "credit",
        "amount": Decimal(amount),
        "currency": "INR",
        "transaction_date": datetime.date(2026, 10, 20),
        "transaction_time": datetime.time(14, 6),
        "counterparty": None,
        "channel": "card",
    }


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
