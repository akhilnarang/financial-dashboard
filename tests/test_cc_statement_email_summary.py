"""Tests for ``process_cc_statement_email_summary`` and the summary-path
wiring in ``parse_email_by_kind``.

Builds ``ParsedEmail`` / ``StatementSummary`` / ``Money`` instances directly
so the tests run against the real parser contract. They exercise:

- The account-selection branches (0, 1, many with/without card_mask)
- Upload row field population (source_kind, empty-string filename/file_path, etc.)
- The password-hint regression fix in ``parse_email_by_kind``
- Idempotency of the ``init_db`` migration
"""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    Card,
    StatementUpload,
)
from financial_dashboard.db.init_db import init_db as _init_db
import financial_dashboard.services.emails as emails_service
from financial_dashboard.services.statements import cc as cc_module
from bank_email_parser.models import Money, ParsedEmail, StatementSummary


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch, tmp_path):
    db_path = tmp_path / "summary-test.sqlite"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()


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
async def test_summary_creates_statement_upload_with_correct_fields(
    session_factory, monkeypatch
):
    acc_id = await _add_cc_account(session_factory)
    parsed = _parsed_with(_default_summary())

    # Avoid the Telegram notification path (no tg_app).
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    # Keep init_payment_tracking out of the way — it uses its own async_session.
    async def _noop(_uid):
        return True

    import financial_dashboard.services.reminders as reminders_mod

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is not None
    assert result["summary_only"] is True
    assert "statement_upload_id" in result

    async with session_factory() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().first()
        assert upload is not None
        assert upload.account_id == acc_id
        assert upload.source_kind == "email_summary"
        assert upload.filename == ""
        assert upload.file_path == ""
        assert upload.status == "parsed"
        assert upload.card_number == "1234"
        assert upload.due_date == "05/05/2026"
        assert upload.total_amount_due == "12,899.94"
        assert upload.minimum_amount_due == "371.94"
        assert upload.parsed_txn_count == 0
        assert upload.matched_count == 0
        assert upload.missing_count == 0
        assert upload.imported_count == 0
        assert upload.reconciliation_data is None


@pytest.mark.anyio
async def test_summary_returns_none_when_no_cc_account(session_factory, monkeypatch):
    parsed = _parsed_with(_default_summary())
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_when_total_amount_missing(session_factory, monkeypatch):
    """Partial ``StatementSummary`` (missing ``total_amount_due``) must not
    produce a phantom upload row. Summary uploads are not retryable, so an
    insert without the dashboard-critical field would be unrecoverable."""
    await _add_cc_account(session_factory)
    summary = _default_summary()
    summary.total_amount_due = None
    parsed = _parsed_with(summary)
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_when_due_date_missing(session_factory, monkeypatch):
    """Same contract: ``due_date`` is load-bearing for the reminder pipeline."""
    await _add_cc_account(session_factory)
    summary = _default_summary()
    summary.due_date = None
    parsed = _parsed_with(summary)
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_to_autopick_with_multiple_accounts_no_card_mask(
    session_factory, monkeypatch, caplog
):
    await _add_cc_account(session_factory, label="OneCard A")
    await _add_cc_account(session_factory, label="OneCard B")
    summary = _default_summary()
    summary.card_mask = None
    parsed = _parsed_with(summary)
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    with caplog.at_level("WARNING"):
        result = await cc_module.process_cc_statement_email_summary(
            "onecard", parsed, email_id=None
        )

    assert result is None
    assert any("multiple CC accounts" in r.message for r in caplog.records)
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.anyio
async def test_summary_refuses_when_multiple_accounts_share_last4(
    session_factory, monkeypatch, caplog
):
    """Two active CC accounts sharing the same last-4 (e.g. physical + virtual
    card, or a re-issued card). Refuse to auto-pick instead of silently
    attaching to the first match."""
    await _add_cc_account(session_factory, label="OneCard A", card_last4="1234")
    await _add_cc_account(session_factory, label="OneCard B", card_last4="1234")
    parsed = _parsed_with(_default_summary())  # card_mask="1234"
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    with caplog.at_level("WARNING"):
        result = await cc_module.process_cc_statement_email_summary(
            "onecard", parsed, email_id=None
        )

    assert result is None
    assert any("ambiguous CC account match" in r.message for r in caplog.records)
    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []


@pytest.mark.parametrize(
    ("accounts", "card_mask", "attaches", "expected_log"),
    [
        # A left-visible BIN denotes no card. Flattened to digits it would
        # read as the suffix of the account ending 1234 — one clean, wrong
        # hit. It selects nothing, and with two accounts there is no sole
        # account to fall back to.
        pytest.param(
            [{"card_last4": "9999"}, {"card_last4": "1234"}],
            "1234XXXXXXXX",
            False,
            None,
            id="bin-only-mask-two-accounts-refused",
        ),
        # One account is not a licence to skip the card check: a readable
        # mask that positively disagrees means this summary is about a card
        # the dashboard does not track.
        pytest.param(
            [{"card_last4": "9012"}],
            "XXXX XXXX XXXX 7788",
            False,
            "disagrees with the only credit_card account",
            id="sole-account-refuted-by-readable-mask",
        ),
        # Some banks print no mask at all; absent data is not a conflict.
        pytest.param(
            [{"card_last4": "9012"}],
            None,
            True,
            None,
            id="absent-mask-attaches-to-sole-account",
        ),
        # The stored side of the same rule: an account recording no card has
        # said nothing, and silence cannot disagree.
        pytest.param(
            [{}],
            "1234",
            True,
            None,
            id="sole-account-recording-no-mask-attaches",
        ),
        # Refuting the only account needs no trailing digits: the BIN shows a
        # digit where the stored mask shows a different one.
        pytest.param(
            [{"account_number": "5100XXXXXXXX9012"}],
            "1234 XXXX XXXX XXXX",
            False,
            "disagrees with the only credit_card account",
            id="left-bin-disagreeing-refused",
        ),
        # ...and a BIN agreeing everywhere the two masks can see each other
        # contradicts nothing.
        pytest.param(
            [{"account_number": "5100XXXXXXXX9012"}],
            "5100 XXXX XXXX XXXX",
            True,
            None,
            id="left-bin-consistent-attaches",
        ),
        # The limit of what refutation can see, pinned so it is not mistaken
        # for a bug later: masks compare right-aligned, so a 12-position
        # BIN's visible digits land against the 16-position stored mask's
        # hidden middle and can contradict nothing.
        pytest.param(
            [{"account_number": "5100XXXXXXXX9012"}],
            "1234XXXXXXXX",
            True,
            None,
            id="short-bin-cannot-reach-stored-digits",
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
            None,
            id="addon-card-keeps-sole-account",
        ),
    ],
)
@pytest.mark.anyio
async def test_summary_card_mask_gates_attachment(
    session_factory, monkeypatch, caplog, accounts, card_mask, attaches, expected_log
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
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    import financial_dashboard.services.reminders as reminders_mod

    async def _noop(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)

    with caplog.at_level("WARNING"):
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
    if expected_log:
        assert any(expected_log in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_summary_picks_matching_account_by_card_mask(
    session_factory, monkeypatch
):
    await _add_cc_account(session_factory, label="OneCard A", card_last4="9999")
    target_id = await _add_cc_account(
        session_factory, label="OneCard B", card_last4="1234"
    )
    parsed = _parsed_with(_default_summary())

    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    import financial_dashboard.services.reminders as reminders_mod

    async def _noop(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)

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
    session_factory, monkeypatch
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

    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    import financial_dashboard.services.reminders as reminders_mod

    async def _noop(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )

    assert result is None
    async with session_factory() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().first()
        assert upload is None


@pytest.mark.anyio
async def test_summary_dedupes_on_reparse(session_factory, monkeypatch):
    """Reprocessing the same summary email (via reparse, or any re-entry of
    the handler) must NOT create a parallel ``StatementUpload`` row. The
    second call should update the existing row in place."""
    acc_id = await _add_cc_account(session_factory)
    parsed = _parsed_with(_default_summary())
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    import financial_dashboard.services.reminders as reminders_mod

    async def _noop(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop)

    first = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )
    assert first is not None
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
        assert rows[0].account_id == acc_id
        # email_id backfilled on the second call.
        assert rows[0].email_id == 42


# ---------------------------------------------------------------------------
# Regression: parse_email_by_kind must extract password_hint for statement kinds
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parse_email_by_kind_threads_password_hint_for_statement_emails(
    monkeypatch,
):
    """Bug fix: previously the HTML parser was skipped for statement kinds,
    silently dropping any ``password_hint`` emitted by the email parser."""

    fake_parsed = SimpleNamespace(
        bank="hdfc",
        email_type="hdfc_cc_statement",
        transaction=None,
        password_hint="DOB in DDMMYYYY",
        statement=None,
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
    fake_parsed = SimpleNamespace(
        bank="hdfc",
        email_type="hdfc_cc_txn",
        transaction=SimpleNamespace(
            direction="debit",
            amount=SimpleNamespace(amount=Decimal("100"), currency="INR"),
            transaction_date=date(2026, 4, 1),
            transaction_time=None,
            counterparty="merchant",
            card_mask="1234",
            account_mask=None,
            reference_number="ref",
            channel="pos",
            balance=None,
            raw_description="",
        ),
        password_hint=None,
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
    fake_parsed = SimpleNamespace(
        bank="onecard",
        email_type="onecard_cc_statement",
        transaction=None,
        password_hint=None,
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


@pytest.mark.anyio
async def test_parse_email_by_kind_distinguishes_handler_exception_from_refusal(
    monkeypatch,
):
    """Regression: when the summary handler *raises* (vs returning None), the
    error surfaced to the email row must reflect the actual exception, not the
    canned 'no matching CC account' refusal message."""
    fake_summary = StatementSummary(
        total_amount_due=Money(amount=Decimal("100.00")),
        due_date=date(2099, 1, 1),
    )
    fake_parsed = SimpleNamespace(
        bank="onecard",
        email_type="onecard_cc_statement",
        transaction=None,
        password_hint=None,
        statement=fake_summary,
    )
    monkeypatch.setattr(emails_service, "parse_email", lambda bank, html: fake_parsed)
    monkeypatch.setattr(
        emails_service, "_extract_html_body", lambda raw: "<html>ignored</html>"
    )
    monkeypatch.setattr(emails_service, "_extract_text_body", lambda raw: "")

    async def _raise(*a, **kw):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(emails_service, "process_cc_statement_email_summary", _raise)

    result = await emails_service.parse_email_by_kind(
        bank="onecard",
        email_kind=None,
        raw_bytes=b"",
        subject="BOBCARD statement",
        source_id=None,
        log_ref="test",
    )

    assert result.stmt_result is None
    assert result.error is not None
    assert "db exploded" in result.error
    # Must NOT mis-report as an account/ambiguity refusal.
    assert "matching CC account" not in result.error
    assert "ambiguous" not in result.error


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_init_db_is_idempotent(tmp_path, monkeypatch):
    """Running init_db twice should be a no-op the second time (no errors,
    and the schema should not accumulate duplicate structures)."""

    db_path = tmp_path / "idempotent.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # Patch load_all_settings and load_merchant_rules — both are called at the
    # tail of init_db but need the real async_session which we aren't using here.
    import financial_dashboard.services.settings as settings_mod
    import financial_dashboard.services.categorization.merchant_rules as mr_mod

    async def _noop_load() -> dict[str, str]:
        return {}

    async def _noop_load_mr() -> None:
        pass

    try:
        monkeypatch.setattr(settings_mod, "load_all_settings", _noop_load)
        monkeypatch.setattr(mr_mod, "load_merchant_rules", _noop_load_mr)
        await _init_db(engine)
        await _init_db(engine)  # second run: must not error
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_init_db_summary_columns_present_and_filename_non_null(
    tmp_path, monkeypatch
):
    """After ``init_db``, ``statement_uploads`` must expose ``source_kind`` and
    ``minimum_amount_due`` columns, while ``filename`` and ``file_path`` remain
    NOT NULL. Summary rows use ``filename=""`` / ``file_path=""`` placeholders
    — see ``process_cc_statement_email_summary``."""

    from sqlalchemy import text as _text

    db_path = tmp_path / "schema.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    import financial_dashboard.services.settings as settings_mod
    import financial_dashboard.services.categorization.merchant_rules as mr_mod

    async def _noop_load() -> dict[str, str]:
        return {}

    async def _noop_load_mr() -> None:
        pass

    try:
        monkeypatch.setattr(settings_mod, "load_all_settings", _noop_load)
        monkeypatch.setattr(mr_mod, "load_merchant_rules", _noop_load_mr)
        await _init_db(engine)

        async with engine.begin() as conn:
            rows = (
                await conn.execute(_text("PRAGMA table_info(statement_uploads)"))
            ).all()
        nullable = {row[1]: row[3] == 0 for row in rows}
        assert nullable["filename"] is False
        assert nullable["file_path"] is False
        assert "source_kind" in nullable
        assert "minimum_amount_due" in nullable
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_init_db_migrates_pre_branch_schema(tmp_path, monkeypatch):
    """Running ``init_db`` against a DB seeded with the pre-branch
    ``statement_uploads`` shape (no ``source_kind`` / ``minimum_amount_due``)
    must add both columns AND backfill existing rows with
    ``source_kind='pdf'`` via the ``DEFAULT 'pdf'`` in the ALTER statement.

    This is the case that matters for real deployments — a fresh
    ``create_all`` test DB never exercises the ALTER path."""
    from sqlalchemy import text as _text

    db_path = tmp_path / "pre_branch.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    import financial_dashboard.services.settings as settings_mod
    import financial_dashboard.services.categorization.merchant_rules as mr_mod

    async def _noop_load() -> dict[str, str]:
        return {}

    async def _noop_load_mr() -> None:
        pass

    try:
        monkeypatch.setattr(settings_mod, "load_all_settings", _noop_load)
        monkeypatch.setattr(mr_mod, "load_merchant_rules", _noop_load_mr)

        # Seed the pre-branch schema by hand — just the rows ``init_db`` looks
        # at. All columns the inline migrations may ADD are intentionally
        # absent; we only ensure the 2 new-in-branch columns are the ones
        # under test. ``settings`` is required by the NACH-marker migration.
        async with engine.begin() as conn:
            await conn.execute(
                _text(
                    "CREATE TABLE accounts (id INTEGER PRIMARY KEY, "
                    "bank TEXT NOT NULL, label TEXT NOT NULL, type TEXT NOT NULL, "
                    "active INTEGER DEFAULT 1)"
                )
            )
            await conn.execute(
                _text(
                    "CREATE TABLE emails (id INTEGER PRIMARY KEY, "
                    "source_id INTEGER, remote_id TEXT, rule_id INTEGER, "
                    "provider TEXT, message_id TEXT, sender TEXT, subject TEXT, "
                    "received_at DATETIME, status TEXT, error TEXT, "
                    "fetched_at DATETIME)"
                )
            )
            await conn.execute(
                _text(
                    "CREATE TABLE statement_uploads ("
                    "id INTEGER PRIMARY KEY, "
                    "account_id INTEGER NOT NULL REFERENCES accounts(id), "
                    "bank TEXT NOT NULL, "
                    "filename TEXT NOT NULL, "
                    "file_path TEXT NOT NULL, "
                    "status TEXT NOT NULL DEFAULT 'parsed', "
                    "card_number TEXT, statement_name TEXT, "
                    "due_date TEXT, total_amount_due TEXT, "
                    "parsed_txn_count INTEGER DEFAULT 0, "
                    "matched_count INTEGER DEFAULT 0, "
                    "missing_count INTEGER DEFAULT 0, "
                    "imported_count INTEGER DEFAULT 0, "
                    "reconciliation_data TEXT, error TEXT, "
                    "created_at DATETIME)"
                )
            )
            await conn.execute(
                _text("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
            )
            # Seed a legacy row. No source_kind — the migration must backfill.
            await conn.execute(
                _text(
                    "INSERT INTO accounts (id, bank, label, type) VALUES "
                    "(1, 'onecard', 'OneCard', 'credit_card')"
                )
            )
            await conn.execute(
                _text(
                    "INSERT INTO statement_uploads "
                    "(id, account_id, bank, filename, file_path, status, "
                    "total_amount_due, due_date) VALUES "
                    "(42, 1, 'onecard', 'stmt.pdf', '/tmp/stmt.pdf', 'parsed', "
                    "'1,000.00', '01/01/2099')"
                )
            )

        await _init_db(engine)

        async with engine.begin() as conn:
            cols = {
                row[1]
                for row in (
                    await conn.execute(_text("PRAGMA table_info(statement_uploads)"))
                ).all()
            }
            assert "source_kind" in cols
            assert "minimum_amount_due" in cols

            row = (
                await conn.execute(
                    _text(
                        "SELECT source_kind, minimum_amount_due, filename, "
                        "total_amount_due FROM statement_uploads WHERE id = 42"
                    )
                )
            ).one()
        source_kind, min_due, filename, total = row
        # DEFAULT 'pdf' backfills the legacy row.
        assert source_kind == "pdf"
        # Newly-added nullable column — existing row gets NULL.
        assert min_due is None
        # Existing data preserved.
        assert filename == "stmt.pdf"
        assert total == "1,000.00"
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_retry_cc_statement_upload_skips_email_summary(
    session_factory, monkeypatch, caplog
):
    """``retry_cc_statement_upload`` must early-return False for summary-only
    uploads — they have no PDF to reparse. Guards against the retry pipeline
    attempting to load an empty ``file_path`` for ``source_kind='email_summary'``
    rows."""
    from financial_dashboard.services.statements import shared as shared_module

    monkeypatch.setattr(shared_module, "async_session", session_factory)

    acc_id = await _add_cc_account(session_factory)
    async with session_factory() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="onecard",
            filename="",
            file_path="",
            source_kind="email_summary",
            status="parsed",
            due_date="05/05/2026",
            total_amount_due="12,899.94",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    with caplog.at_level("INFO"):
        ok = await shared_module.retry_cc_statement_upload(upload_id, password="x")

    assert ok is False
    assert any("email_summary" in r.message for r in caplog.records)
