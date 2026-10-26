"""Receivable payment detection and its Telegram tap. Names and amounts are synthetic."""

from collections.abc import AsyncIterator, Mapping, Sequence
from decimal import Decimal
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram import InlineKeyboardMarkup
from telegram.ext import Application

from financial_dashboard.db.models import (
    PersonIdentifier,
    Receivable,
    ReceivableLink,
    ReceivableMatchPrompt,
    Transaction,
)
from financial_dashboard.services import receivable_matches, telegram
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services.receivable_matches import (
    run_receivable_match_sweep,
)
from financial_dashboard.schemas.receivables import ReceivableRead
from financial_dashboard.services.receivables import receivable_read
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio

TxnFields = Mapping[str, Decimal | str | None]


class Rcv(NamedTuple):
    maker: async_sessionmaker[AsyncSession]
    sent: list[tuple[str, InlineKeyboardMarkup | None]]


class Tap(NamedTuple):
    query: MagicMock
    update: SimpleNamespace


@pytest.fixture
async def rcv(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rcv]:
    """Configure Telegram and record each prompt that reaches the bot."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(receivable_matches, "async_session", maker)
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    settings_service._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "token",
            "telegram.chat_id": "7",
            "telegram.assistant_enabled": "true",
        }
    )
    sent: list[tuple[str, InlineKeyboardMarkup | None]] = []

    async def fake_send(
        app: Application,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> SimpleNamespace:
        sent.append((text, reply_markup))
        return SimpleNamespace(message_id=100 + len(sent))

    monkeypatch.setattr(telegram, "_send_with_retry", fake_send)
    yield Rcv(maker, sent)
    await engine.dispose()
    holder.close()


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    receivables: Sequence[tuple[str, ...]],
    txn: TxnFields,
    identifiers: Sequence[tuple[str, str, str]] = (),
) -> int:
    """Seed receivables, identifiers and one row. Return the row id."""
    async with maker() as s:
        row = Transaction(
            bank="testbank", email_type="testbank_alert", currency="INR", **txn
        )
        # A settled receivable already takes the row as its payment.
        s.add_all(
            Receivable(
                kind=kind,
                person=person,
                amount=Decimal(amount),
                status=status[0],
                links=[ReceivableLink(transaction=row, role="payment")],
            )
            if status
            else Receivable(kind=kind, person=person, amount=Decimal(amount))
            for kind, person, amount, *status in receivables
        )
        s.add_all(
            PersonIdentifier(person=person, kind=kind, value=value)
            for person, kind, value in identifiers
        )
        s.add(row)
        await s.commit()
        return row.id


def _credit(amount: str, counterparty: str, raw: str | None = None) -> TxnFields:
    return dict(
        direction="credit",
        amount=Decimal(amount),
        counterparty=counterparty,
        raw_description=raw,
    )


@pytest.mark.parametrize(
    ("receivables", "identifiers", "txn", "offered"),
    [
        # A stored VPA is strong. It ranks before a longer name match of
        # another person, and both are offered.
        (
            [
                ("receivable", "Gamma Delta Epsilon", "500"),
                ("receivable", "Friend Alpha", "500"),
            ],
            [("friend alpha", "vpa", "alpha.f@okbank")],
            _credit("500", "GAMMA DELTA EPSILON", "UPI/CR/alpha.f@okbank/GDE"),
            ["Friend Alpha", "Gamma Delta Epsilon"],
        ),
        # A dash-joined narration glues the name to a stored VPA.
        (
            [("receivable", "Friend Alpha", "500")],
            [("friend alpha", "vpa", "alpha.f@okbank")],
            _credit(
                "500",
                "UPI TRANSFER",
                "UPI-FRIEND ALPHA-alpha.f@okbank-OKBK0000001-123456789012-NOTE",
            ),
            ["Friend Alpha"],
        ),
        # A VPA stored with a narration prefix still matches a plain one.
        (
            [("receivable", "Friend Alpha", "500")],
            [("friend alpha", "vpa", "upi-alpha.f@okbank")],
            _credit("500", "UPI TRANSFER", "UPI/CR/alpha.f@okbank/REF000002"),
            ["Friend Alpha"],
        ),
        # A short stored name inside a full name does not hide that name.
        (
            [
                ("receivable", "Ravi Kumar", "500"),
                ("receivable", "Ravindra Patil", "500"),
            ],
            [("ravi kumar", "counterparty", "ravi")],
            _credit("500", "RAVINDRA PATIL"),
            ["Ravindra Patil"],
        ),
        # A joint account glues two names together. Both are offered.
        (
            [("receivable", "Ravi Kumar", "500"), ("receivable", "Priya Shah", "500")],
            [],
            _credit("500", "RAVIKUMARPRIYASHAH"),
            ["Ravi Kumar", "Priya Shah"],
        ),
        # A learned identifier of one holder does not hide the other holder.
        (
            [("receivable", "Priya Shah", "500"), ("receivable", "Ravi Kumar", "500")],
            [("ravi kumar", "counterparty", "ravikumar")],
            _credit("500", "RAVIKUMARPRIYASHAH"),
            ["Ravi Kumar", "Priya Shah"],
        ),
        # One name sits inside a longer full name. The full name wins.
        (
            [
                ("receivable", "Asha Varma", "500"),
                ("receivable", "Asha Varma Nair", "500"),
            ],
            [],
            _credit("500", "ASHA VARMA NAIR"),
            ["Asha Varma Nair"],
        ),
        # A name match is a fallback only for a person with no identifiers.
        (
            [("receivable", "Ravi Kumar", "500")],
            [("ravi kumar", "vpa", "ravi.k@okbank")],
            _credit("500", "RAVI KUMAR", "UPI/CR/other.p@okbank"),
            [],
        ),
        # A row that pays a receivable is not offered again.
        (
            [
                ("receivable", "Ravi Kumar", "500"),
                ("receivable", "Ravi Kumar", "500", "settled"),
            ],
            [],
            _credit("500", "RAVI KUMAR"),
            [],
        ),
        # A partial payment is not offered.
        ([("receivable", "Ravi Kumar", "500")], [], _credit("300", "RAVI KUMAR"), []),
        # A payable takes a debit, and a receivable does not.
        (
            [("payable", "Ravi Kumar", "500")],
            [],
            {**_credit("500", "RAVI KUMAR"), "direction": "debit"},
            ["Ravi Kumar"],
        ),
        (
            [("receivable", "Ravi Kumar", "500")],
            [],
            {**_credit("500", "RAVI KUMAR"), "direction": "debit"},
            [],
        ),
        # More than three matches get no prompt.
        (
            [("receivable", "Ravi Kumar", "500")] * 4,
            [],
            _credit("500", "RAVI KUMAR"),
            [],
        ),
    ],
    ids=[
        "strong",
        "dashed",
        "prefixed_stored",
        "short_inside",
        "joint",
        "joint_learned",
        "inside",
        "known",
        "linked",
        "partial",
        "payable",
        "debit",
        "crowd",
    ],
)
async def test_sweep_offers_exact_matches_once(
    rcv, receivables, identifiers, txn, offered
):
    maker, sent = rcv
    txn_id = await _seed(maker, receivables, txn, identifiers)

    first = await run_receivable_match_sweep()
    again = await run_receivable_match_sweep()

    async with maker() as s:
        prompts = (await s.scalars(select(ReceivableMatchPrompt))).all()
        persons = [
            (await s.get(Receivable, rid)).person
            for prompt in prompts
            for rid in prompt.candidate_ids
        ]
    assert (first, again) == (int(bool(offered)), 0)
    assert persons == offered
    assert [p.transaction_id for p in prompts] == ([txn_id] if offered else [])
    if offered:
        text, markup = sent[0]
        assert f"#{txn_id}" in text
        buttons = [button.text for row in markup.inline_keyboard for button in row]
        assert buttons == [
            f"Link to {person} #{rid}"
            for person, rid in zip(offered, prompts[0].candidate_ids)
        ] + ["Not this"]


def _tap(data: str, chat_id: int = 7, message_id: int = 101) -> Tap:
    """Build a callback tap on a prompt message, as Telegram sends it."""
    query = MagicMock()
    query.data = data
    query.message.chat.id = chat_id
    query.message.message_id = message_id
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return Tap(query, SimpleNamespace(callback_query=query))


@pytest.mark.parametrize(
    "action", ["link", "dismiss", "stale", "voided", "renamed_row", "old_message"]
)
async def test_tap_links_only_once_and_never_settles(rcv, action):
    maker, sent = rcv
    txn_id = await _seed(
        maker,
        [("receivable", "Ravi Kumar", "500")],
        _credit(
            "500",
            "RAVI KUMAR",
            # The 12-digit reference is not a phone, so the one phone is taught.
            "UPI-RAVI KUMAR-ravi.k@okbank-OKBK0000001-9876500002-919876500004",
        ),
    )
    assert await run_receivable_match_sweep() == 1
    link, dismiss = [b for row in sent[0][1].inline_keyboard for b in row]
    if action == "stale":
        # Another payment lands before the tap, so the amount no longer fits.
        async with maker() as s:
            other = Transaction(
                bank="testbank",
                email_type="testbank_alert",
                direction="credit",
                amount=Decimal("100"),
            )
            s.add(other)
            await s.flush()
            s.add(
                ReceivableLink(receivable_id=1, transaction_id=other.id, role="payment")
            )
            await s.commit()
    if action == "voided":
        # The bank voids the row before the tap.
        async with maker() as s:
            (await s.get(Transaction, txn_id)).voided = True
            await s.commit()
    if action == "renamed_row":
        # A reparse names another sender on the row.
        async with maker() as s:
            (await s.get(Transaction, txn_id)).counterparty = "SOMEONE ELSE"
            await s.commit()
    # SQLite can give a deleted prompt id to a new prompt. An old message
    # with that id must not act on it.
    message_id = 999 if action == "old_message" else 101
    data = (dismiss if action == "dismiss" else link).callback_data

    stranger, stranger_update = _tap(data, chat_id=999)
    await telegram._handle_callback(stranger_update, MagicMock())
    first, first_update = _tap(data, message_id=message_id)
    await telegram._handle_callback(first_update, MagicMock())
    second, second_update = _tap(link.callback_data, message_id=message_id)
    await telegram._handle_callback(second_update, MagicMock())

    async with maker() as s:
        read = receivable_read(await s.get(Receivable, 1))
        learned = (
            await s.scalars(
                select(PersonIdentifier.value).order_by(PersonIdentifier.value)
            )
        ).all()
    stranger.answer.assert_awaited_once_with("Unauthorized")
    assert read.status == "open"
    if action == "link":
        first.edit_message_text.assert_awaited_once()
        assert "fully paid" in first.edit_message_text.await_args.args[0]
        second.answer.assert_awaited_once_with("Already handled")
        assert (read.payment_transaction_ids, read.outstanding) == (
            [txn_id],
            Decimal("0.00"),
        )
        # A dash-joined narration teaches the VPA without the name prefix.
        assert learned == ["9876500002", "ravi.k@okbank", "ravikumar"]
        # A later row with a plain narration matches the learned VPA.
        await _seed(
            maker,
            [("receivable", "Ravi Kumar", "300")],
            _credit("300", "UNKNOWN SENDER", "UPI/CR/ravi.k@okbank/REF000001"),
        )
        assert await run_receivable_match_sweep() == 1
    else:
        assert txn_id not in read.payment_transaction_ids
        assert learned == []
    if action == "old_message":
        first.answer.assert_awaited_once_with("This button is stale")
    if action in ("stale", "voided", "renamed_row"):
        first.answer.assert_awaited_once_with(
            "This receivable changed. Link the row on the page."
        )
        second.edit_message_text.assert_not_awaited()


async def test_sweep_skips_a_row_that_fails_and_prompts_the_rest(rcv, monkeypatch):
    maker, sent = rcv
    first = await _seed(
        maker, [("receivable", "Ravi Kumar", "500")], _credit("500", "RAVI KUMAR")
    )
    second = await _seed(
        maker, [("receivable", "Priya Shah", "700")], _credit("700", "PRIYA SHAH")
    )
    queue = receivable_matches._queue_prompt

    async def racing(
        session: AsyncSession,
        txn: Transaction,
        matches: Sequence[ReceivableRead],
        chat_id: int,
    ) -> int:
        if txn.id == first:
            # Another sweep prompted this row a moment ago.
            session.add(ReceivableMatchPrompt(transaction_id=txn.id, candidate_ids=[]))
            await session.flush()
        return await queue(session, txn, matches, chat_id)

    monkeypatch.setattr(receivable_matches, "_queue_prompt", racing)

    assert await run_receivable_match_sweep() == 1
    async with maker() as s:
        prompted = (await s.scalars(select(ReceivableMatchPrompt.transaction_id))).all()
    assert prompted == [second]
    assert len(sent) == 1
