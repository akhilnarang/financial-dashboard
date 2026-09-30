"""Tests for ``process_cc_statement_email_summary`` and the summary-path
wiring in ``parse_email_by_kind``.

Builds ``ParsedEmail`` / ``StatementSummary`` / ``Money`` instances directly
so the tests run against the real parser contract. They exercise:

- The account-selection branches (0, 1, many with/without card_mask)
- Upload row field population and reparse dedup
- The password-hint regression fix in ``parse_email_by_kind``
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db import (
    Account,
    Card,
    StatementUpload,
)
import financial_dashboard.services.emails as emails_service
import financial_dashboard.services.reminders as reminders_mod
from financial_dashboard.services.statements import cc as cc_module
from bank_email_parser.models import (
    Money,
    ParsedEmail,
    StatementSummary,
    TransactionAlert,
)
from tests.conftest import new_test_engine


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    # Keep Telegram and payment tracking out of the way.
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    async def _noop(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)
    yield maker
    await engine.dispose()
    holder.close()


def _default_summary() -> StatementSummary:
    return StatementSummary(
        total_amount_due=Money(amount=Decimal("12899.94")),
        minimum_amount_due=Money(amount=Decimal("371.94")),
        due_date=date(2026, 5, 5),
        card_mask="1234",
    )


def _parsed_with(summary: StatementSummary | None) -> ParsedEmail:
    return ParsedEmail(
        email_type="onecard_cc_statement",
        bank="onecard",
        statement=summary,
    )


async def _add_cc_account(
    maker,
    *,
    bank: str = "onecard",
    label: str = "OneCard",
    account_number: str | None = None,
    active: bool = True,
    card_last4: str | None = None,
) -> int:
    async with maker() as session:
        acc = Account(
            bank=bank,
            label=label,
            type="credit_card",
            account_number=account_number,
            active=active,
        )
        session.add(acc)
        await session.flush()
        if card_last4:
            session.add(Card(account_id=acc.id, card_mask=card_last4, is_primary=True))
        await session.commit()
        return acc.id


@pytest.mark.anyio
async def test_summary_returns_none_when_no_cc_account(session_factory):
    parsed = _parsed_with(_default_summary())

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
@pytest.mark.parametrize("missing", ["total_amount_due", "due_date"])
async def test_summary_refuses_when_required_field_missing(session_factory, missing):
    """A partial ``StatementSummary`` must not produce a phantom upload row.
    Summary uploads are not retryable. The reminder pipeline needs both fields."""
    await _add_cc_account(session_factory)
    summary = _default_summary()
    setattr(summary, missing, None)
    parsed = _parsed_with(summary)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_to_autopick_with_multiple_accounts_no_card_mask(
    session_factory,
):
    await _add_cc_account(session_factory, label="OneCard A")
    await _add_cc_account(session_factory, label="OneCard B")
    summary = _default_summary()
    summary.card_mask = None
    parsed = _parsed_with(summary)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_when_multiple_accounts_share_last4(session_factory):
    """Two active CC accounts sharing the same last-4 (e.g. physical + virtual
    card, or a re-issued card). Refuse to auto-pick instead of silently
    attaching to the first match."""
    await _add_cc_account(session_factory, label="OneCard A", card_last4="1234")
    await _add_cc_account(session_factory, label="OneCard B", card_last4="1234")
    parsed = _parsed_with(_default_summary())  # card_mask="1234"

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.parametrize(
    ("accounts", "card_mask", "attaches"),
    [
        # A left-visible BIN denotes no card. Flattened to digits it would
        # read as the suffix of the account ending 1234 — one clean, wrong
        # hit. It selects nothing, and with two accounts there is no sole
        # account to fall back to.
        pytest.param(
            [{"card_last4": "9999"}, {"card_last4": "1234"}],
            "1234XXXXXXXX",
            False,
            id="bin-only-mask-two-accounts-refused",
        ),
        # Some banks print no mask at all; absent data is not a conflict.
        pytest.param(
            [{"card_last4": "9012"}],
            None,
            True,
            id="absent-mask-attaches-to-sole-account",
        ),
        # One account is not a licence to skip the card check. Refuting it
        # needs no trailing digits: the BIN shows a digit where the stored
        # mask shows a different one.
        pytest.param(
            [{"account_number": "5100XXXXXXXX9012"}],
            "1234 XXXX XXXX XXXX",
            False,
            id="left-bin-disagreeing-refused",
        ),
        # An account is refuted only when NONE of its cards can be the one
        # named — otherwise every add-on statement would be thrown away as a
        # conflict with the primary.
        pytest.param(
            [
                {
                    "account_number": "5100XXXXXXXX9012",
                    "card_last4": "4111XXXXXXXX7788",
                }
            ],
            "XXXX XXXX XXXX 7788",
            True,
            id="addon-card-keeps-sole-account",
        ),
    ],
)
@pytest.mark.anyio
async def test_summary_card_mask_gates_attachment(
    session_factory, accounts, card_mask, attaches
):
    """The card-mask gate on the summary path, in both directions.

    With one account on the bank the question is refutation — attach unless
    the mask positively disagrees; with several it is selection, which needs
    trailing visible digits. Either way a refused summary leaves no upload
    row, and an attached one lands on the first (sole) account.
    """
    first_id = None
    for idx, kwargs in enumerate(accounts):
        acc_id = await _add_cc_account(
            session_factory, label=f"OneCard {idx}", **kwargs
        )
        first_id = first_id if first_id is not None else acc_id
    summary = _default_summary()
    summary.card_mask = card_mask
    parsed = _parsed_with(summary)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    async with session_factory() as session:
        uploads = (await session.execute(select(StatementUpload))).scalars().all()

    if attaches:
        assert result is not None
        assert [upload.account_id for upload in uploads] == [first_id]
    else:
        assert result is None
        assert uploads == []


@pytest.mark.anyio
async def test_summary_picks_matching_account_by_card_mask(session_factory):
    await _add_cc_account(session_factory, label="OneCard A", card_last4="9999")
    target_id = await _add_cc_account(
        session_factory, label="OneCard B", card_last4="1234"
    )
    parsed = _parsed_with(_default_summary())

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is not None
    async with session_factory() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().first()
        assert upload is not None
        assert upload.account_id == target_id


@pytest.mark.anyio
async def test_summary_refuses_when_only_match_is_a_bin_only_stored_mask(
    session_factory,
):
    """A stored mask carrying no trailing visible digit identifies no card.

    Account A holds a real card that does not match the statement; account B
    holds a BIN-only mask (5100XXXXXXXX). Because mask_matches right-aligns,
    B's wildcard suffix lands over the statement's real digits and matches
    every card of that issuer — so without a trailing-digit gate the summary
    would attach to B, an account it has nothing to do with. It must refuse.
    """
    await _add_cc_account(session_factory, label="OneCard A", card_last4="9999")
    await _add_cc_account(session_factory, label="OneCard B", card_last4="5100XXXXXXXX")
    parsed = _parsed_with(_default_summary())  # statement card_mask "1234"

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().first()
        assert upload is None


@pytest.mark.anyio
async def test_summary_creates_one_upload_and_dedupes_on_reparse(session_factory):
    """The first call stores a summary-only upload. Reprocessing the same email
    must update that row in place, not create a parallel one."""
    acc_id = await _add_cc_account(session_factory)
    parsed = _parsed_with(_default_summary())

    first = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )
    assert first is not None
    assert first["summary_only"] is True
    first_id = first["statement_upload_id"]

    # Second call with the exact same payload — should update, not insert.
    second = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=42
    )
    assert second is not None
    assert second["statement_upload_id"] == first_id

    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert len(rows) == 1
        upload = rows[0]
        assert upload.account_id == acc_id
        # email_id backfilled on the second call.
        assert upload.email_id == 42
        assert upload.source_kind == "email_summary"
        assert upload.status == "parsed"
        assert upload.card_number == "1234"
        assert upload.due_date == "05/05/2026"
        assert upload.total_amount_due == "12,899.94"
        assert upload.minimum_amount_due == "371.94"
        assert upload.imported_count == 0


# ---------------------------------------------------------------------------
# Regression: parse_email_by_kind must extract password_hint for statement kinds
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parse_email_by_kind_threads_password_hint_for_statement_emails(
    monkeypatch,
):
    """Bug fix: previously the HTML parser was skipped for statement kinds,
    silently dropping any ``password_hint`` emitted by the email parser."""

    fake_parsed = ParsedEmail(
        bank="hdfc",
        email_type="hdfc_cc_statement",
        password_hint="DOB in DDMMYYYY",
    )

    monkeypatch.setattr(
        emails_service,
        "parse_email",
        lambda bank, html: fake_parsed,
    )
    monkeypatch.setattr(
        emails_service, "_extract_html_body", lambda raw: "<html>ignored</html>"
    )
    monkeypatch.setattr(emails_service, "_extract_text_body", lambda raw: "")

    # Short-circuit the PDF pipelines — we only care about password_hint here.
    async def _no_stmt(*a, **kw):
        return None

    monkeypatch.setattr(emails_service, "process_statement_email", _no_stmt)
    monkeypatch.setattr(emails_service, "process_bank_statement_email", _no_stmt)

    result = await emails_service.parse_email_by_kind(
        bank="hdfc",
        email_kind="cc_statement",
        raw_bytes=b"",
        subject="Your statement is ready",
        source_id=None,
        log_ref="test",
    )

    assert result.txn_data is None
    assert result.stmt_result is None
    assert result.password_hint == "DOB in DDMMYYYY"
    # ``error`` is allowed to be set (statement path returned nothing), what
    # matters is that password_hint survived.


@pytest.mark.anyio
async def test_parse_email_by_kind_transaction_does_not_route_to_summary(monkeypatch):
    """Regression: a ``TRANSACTION``-kind email must NOT be routed to the
    statement-summary handler even if the parser attaches a ``statement``
    field. Summary routing is only for CC_STATEMENT / STATEMENT / None."""

    fake_summary = StatementSummary(card_mask="1234")
    fake_parsed = ParsedEmail(
        bank="hdfc",
        email_type="hdfc_cc_txn",
        transaction=TransactionAlert(
            direction="debit",
            amount=Money(amount=Decimal("100")),
            transaction_date=date(2026, 4, 1),
            counterparty="merchant",
            card_mask="1234",
            reference_number="ref",
            channel="pos",
        ),
        statement=fake_summary,
    )

    monkeypatch.setattr(emails_service, "parse_email", lambda bank, html: fake_parsed)
    monkeypatch.setattr(
        emails_service, "_extract_html_body", lambda raw: "<html>ignored</html>"
    )
    monkeypatch.setattr(emails_service, "_extract_text_body", lambda raw: "")

    summary_calls: list[tuple] = []

    async def _track_summary(*a, **kw):
        summary_calls.append((a, kw))
        return {"statement_upload_id": 1, "summary_only": True}

    monkeypatch.setattr(
        emails_service, "process_cc_statement_email_summary", _track_summary
    )

    result = await emails_service.parse_email_by_kind(
        bank="hdfc",
        email_kind="transaction",
        raw_bytes=b"",
        subject="Transaction alert",
        source_id=None,
        log_ref="test",
    )

    # Summary handler must never be invoked for a TRANSACTION rule.
    assert summary_calls == []
    # Transaction data should pass through.
    assert result.txn_data is not None
    assert result.stmt_result is None


@pytest.mark.anyio
async def test_parse_email_by_kind_surfaces_error_when_summary_handler_refuses(
    monkeypatch,
):
    """Regression: when the parser emits a statement summary but the summary
    handler returns None (no matching CC account / ambiguous match), the
    email must be reported as ``failed``, not silently ``skipped``.

    Previously the error was only set for ``email_kind in _STATEMENT_KINDS``,
    so a rule with ``email_kind=None`` that happened to match a summary email
    would downgrade to ``skipped`` on refusal.
    """
    fake_summary = StatementSummary(
        total_amount_due=Money(amount=Decimal("100.00")),
        due_date=date(2099, 1, 1),
        card_mask=None,
    )
    fake_parsed = ParsedEmail(
        bank="onecard",
        email_type="onecard_cc_statement",
        statement=fake_summary,
    )

    monkeypatch.setattr(emails_service, "parse_email", lambda bank, html: fake_parsed)
    monkeypatch.setattr(
        emails_service, "_extract_html_body", lambda raw: "<html>ignored</html>"
    )
    monkeypatch.setattr(emails_service, "_extract_text_body", lambda raw: "")

    async def _refuse(*a, **kw):
        return None

    monkeypatch.setattr(emails_service, "process_cc_statement_email_summary", _refuse)

    # email_kind=None is the case that previously silently skipped.
    result = await emails_service.parse_email_by_kind(
        bank="onecard",
        email_kind=None,
        raw_bytes=b"",
        subject="Your BOBCARD One Credit Card statement",
        source_id=None,
        log_ref="test",
    )

    assert result.stmt_result is None
    assert result.error is not None
    assert "summary" in result.error.lower()
