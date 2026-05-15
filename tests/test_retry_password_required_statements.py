"""Tests for ``retry_password_required_statements``.

The behaviour under test is the outer coordinator: it collects all uploads in
``password_required`` state for an account, invokes the per-upload retry
helper for each, and returns a count of successes/failures. The per-upload
helpers themselves (``_retry_cc_statement_upload`` /
``_retry_bank_statement_upload``) are exercised by the manual retry endpoints
in production and would need a full PDF fixture to test in isolation, so we
mock them here and focus on the dispatch logic."""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.main import create_app
import financial_dashboard.core.deps as core_deps
from financial_dashboard.db import (
    Account,
    Base,
    BankStatementUpload,
    StatementUpload,
)
from financial_dashboard.services import accounts as accounts_module
from financial_dashboard.services.statements import dates as dates_module
from financial_dashboard.services.statements import shared as statements_shared
from financial_dashboard.web import bank_statements as bank_routes
from financial_dashboard.web import statements as cc_routes


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    """Swap request/session factories for an in-memory DB."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(statements_shared, "async_session", maker)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()


async def _seed_account_with_uploads(
    maker, cc_statuses: list[str], bank_statuses: list[str]
) -> int:
    async with maker() as session:
        account = Account(bank="HDFC", label="HDFC Credit Card", type="credit_card")
        session.add(account)
        await session.flush()
        for idx, status in enumerate(cc_statuses):
            session.add(
                StatementUpload(
                    account_id=account.id,
                    bank="HDFC",
                    filename=f"cc_{idx}.pdf",
                    file_path=f"/tmp/cc_{idx}.pdf",
                    status=status,
                )
            )
        for idx, status in enumerate(bank_statuses):
            session.add(
                BankStatementUpload(
                    account_id=account.id,
                    bank="HDFC",
                    filename=f"bank_{idx}.pdf",
                    file_path=f"/tmp/bank_{idx}.pdf",
                    status=status,
                )
            )
        await session.commit()
        return account.id


@pytest.mark.anyio
async def test_only_password_required_uploads_are_retried(session_factory):
    """Uploads in other statuses (parsed, parse_error, imported) must be
    skipped — only password_required is in scope."""
    account_id = await _seed_account_with_uploads(
        session_factory,
        cc_statuses=["password_required", "parsed", "parse_error"],
        bank_statuses=["password_required", "imported"],
    )

    cc_helper = AsyncMock(return_value=True)
    bank_helper = AsyncMock(return_value=True)
    async with session_factory() as session:
        result = await accounts_module.retry_password_required_statements(
            session,
            account_id,
            "secret",
            retry_cc_upload=cc_helper,
            retry_bank_upload=bank_helper,
        )

    assert cc_helper.await_count == 1
    assert bank_helper.await_count == 1
    assert result == {
        "cc_retried": 1,
        "bank_retried": 1,
        "cc_failed": 0,
        "bank_failed": 0,
    }


@pytest.mark.anyio
async def test_helper_returning_false_counts_as_failure(session_factory):
    """When the per-upload helper reports failure (e.g. wrong password), it
    should increment *_failed, not *_retried."""
    account_id = await _seed_account_with_uploads(
        session_factory,
        cc_statuses=["password_required", "password_required"],
        bank_statuses=["password_required"],
    )

    cc_helper = AsyncMock(side_effect=[True, False])
    bank_helper = AsyncMock(return_value=False)
    async with session_factory() as session:
        result = await accounts_module.retry_password_required_statements(
            session,
            account_id,
            "wrong",
            retry_cc_upload=cc_helper,
            retry_bank_upload=bank_helper,
        )

    assert result == {
        "cc_retried": 1,
        "bank_retried": 0,
        "cc_failed": 1,
        "bank_failed": 1,
    }


@pytest.mark.anyio
async def test_helper_exception_is_isolated(session_factory):
    """A helper raising shouldn't abort the whole loop — the remaining uploads
    must still be attempted and the raising one counted as failed."""
    account_id = await _seed_account_with_uploads(
        session_factory,
        cc_statuses=["password_required", "password_required"],
        bank_statuses=[],
    )

    cc_helper = AsyncMock(side_effect=[RuntimeError("boom"), True])
    async with session_factory() as session:
        result = await accounts_module.retry_password_required_statements(
            session,
            account_id,
            "secret",
            retry_cc_upload=cc_helper,
        )

    assert cc_helper.await_count == 2
    assert result["cc_retried"] == 1
    assert result["cc_failed"] == 1


@pytest.mark.anyio
async def test_empty_account_returns_zero_counts(session_factory):
    """An account with no password_required uploads should complete quickly
    with all-zero counts and never touch the helpers."""
    account_id = await _seed_account_with_uploads(
        session_factory, cc_statuses=["parsed"], bank_statuses=["imported"]
    )

    cc_helper = AsyncMock()
    bank_helper = AsyncMock()
    async with session_factory() as session:
        result = await accounts_module.retry_password_required_statements(
            session,
            account_id,
            "secret",
            retry_cc_upload=cc_helper,
            retry_bank_upload=bank_helper,
        )

    cc_helper.assert_not_awaited()
    bank_helper.assert_not_awaited()
    assert result == {
        "cc_retried": 0,
        "bank_retried": 0,
        "cc_failed": 0,
        "bank_failed": 0,
    }


# ---------------------------------------------------------------------------
# Manual retry endpoints: password-save ordering
# ---------------------------------------------------------------------------
#
# The manual /statements/{id}/retry and /statements/bank/{id}/retry endpoints
# accept a ``save_password=1`` form field. The expected behaviour is:
#   - retry succeeds + save_password=1  → password is saved
#   - retry fails    + save_password=1  → password is NOT saved (so a wrong
#                                          guess doesn't overwrite a good one)


async def _seed_single_upload(maker, kind: str, status: str) -> tuple[int, int]:
    """Create one account + one upload of the given kind, return (upload_id, account_id)."""
    async with maker() as session:
        account = Account(bank="HDFC", label="HDFC Credit Card", type="credit_card")
        session.add(account)
        await session.flush()
        model = StatementUpload if kind == "cc" else BankStatementUpload
        upload = model(
            account_id=account.id,
            bank="HDFC",
            filename=f"{kind}.pdf",
            file_path=f"/tmp/{kind}.pdf",
            status=status,
        )
        session.add(upload)
        await session.commit()
        return upload.id, account.id


async def _get_account_password(maker, account_id: int) -> str | None:
    async with maker() as session:
        account = await session.get(Account, account_id)
        return account.statement_password if account else None


def _make_helper_that_flips_status(maker, model, result: bool):
    """Return an AsyncMock that flips the target upload's status to 'parsed'
    on success — matching what the real ``_retry_*_statement_upload`` does.

    Without this the bulk-retry coordinator (triggered after a successful
    manual retry with save_password=1) would re-query the DB, still see the
    row in 'password_required', and call the helper a second time."""

    async def _side_effect(upload_id, password):
        if result:
            async with maker() as session:
                if upload := await session.get(model, upload_id):
                    upload.status = "parsed"
                    await session.commit()
        return result

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.anyio
async def test_cc_retry_saves_password_only_on_success(session_factory):
    upload_id, account_id = await _seed_single_upload(
        session_factory, "cc", "password_required"
    )
    from httpx import ASGITransport, AsyncClient

    helper = _make_helper_that_flips_status(
        session_factory, StatementUpload, result=True
    )
    with patch.object(cc_routes, "retry_cc_statement_upload", helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/{upload_id}/retry",
                data={"password": "rightpw", "save_password": "1"},
            )

    assert resp.status_code == 303
    helper.assert_awaited_once_with(upload_id, "rightpw")
    assert await _get_account_password(session_factory, account_id) is not None


@pytest.mark.anyio
async def test_cc_retry_does_not_save_password_on_failure(session_factory):
    upload_id, account_id = await _seed_single_upload(
        session_factory, "cc", "password_required"
    )
    from httpx import ASGITransport, AsyncClient

    helper = AsyncMock(return_value=False)
    with patch.object(cc_routes, "retry_cc_statement_upload", helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/{upload_id}/retry",
                data={"password": "wrongpw", "save_password": "1"},
            )

    assert resp.status_code == 303
    helper.assert_awaited_once_with(upload_id, "wrongpw")
    assert await _get_account_password(session_factory, account_id) is None


@pytest.mark.anyio
async def test_bank_retry_saves_password_only_on_success(session_factory):
    upload_id, account_id = await _seed_single_upload(
        session_factory, "bank", "password_required"
    )
    from httpx import ASGITransport, AsyncClient

    helper = _make_helper_that_flips_status(
        session_factory, BankStatementUpload, result=True
    )
    with patch.object(bank_routes, "retry_bank_statement_upload", helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/bank/{upload_id}/retry",
                data={"password": "rightpw", "save_password": "1"},
            )

    assert resp.status_code == 303
    helper.assert_awaited_once_with(upload_id, "rightpw")
    assert await _get_account_password(session_factory, account_id) is not None


@pytest.mark.anyio
async def test_cc_retry_with_save_password_unlocks_siblings(session_factory):
    """Saving a password via the manual retry should unlock every other
    password_required statement on the same account — not just the one the
    user clicked."""
    from httpx import ASGITransport, AsyncClient

    # Seed: one CC upload we'll retry directly + two sibling uploads stuck
    # in password_required (one CC, one bank).
    clicked_id, account_id = await _seed_single_upload(
        session_factory, "cc", "password_required"
    )
    async with session_factory() as session:
        session.add_all(
            [
                StatementUpload(
                    account_id=account_id,
                    bank="HDFC",
                    filename="cc_sibling.pdf",
                    file_path="/tmp/cc_sibling.pdf",
                    status="password_required",
                ),
                BankStatementUpload(
                    account_id=account_id,
                    bank="HDFC",
                    filename="bank_sibling.pdf",
                    file_path="/tmp/bank_sibling.pdf",
                    status="password_required",
                ),
            ]
        )
        await session.commit()

    cc_helper = _make_helper_that_flips_status(
        session_factory, StatementUpload, result=True
    )
    bank_helper = _make_helper_that_flips_status(
        session_factory, BankStatementUpload, result=True
    )
    with (
        patch.object(cc_routes, "retry_cc_statement_upload", cc_helper),
        patch.object(cc_routes, "retry_bank_statement_upload", bank_helper),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/{clicked_id}/retry",
                data={"password": "goodpw", "save_password": "1"},
            )

    assert resp.status_code == 303
    # Clicked upload + CC sibling each trigger one CC helper call.
    assert cc_helper.await_count == 2
    assert bank_helper.await_count == 1
    assert await _get_account_password(session_factory, account_id) is not None


@pytest.mark.anyio
async def test_cc_retry_without_save_password_does_not_unlock_siblings(session_factory):
    """Without save_password=1, only the clicked upload should be retried —
    sibling password_required statements stay untouched."""
    from httpx import ASGITransport, AsyncClient

    clicked_id, account_id = await _seed_single_upload(
        session_factory, "cc", "password_required"
    )
    async with session_factory() as session:
        session.add(
            StatementUpload(
                account_id=account_id,
                bank="HDFC",
                filename="cc_sibling.pdf",
                file_path="/tmp/cc_sibling.pdf",
                status="password_required",
            )
        )
        await session.commit()

    cc_helper = AsyncMock(return_value=True)
    with patch.object(cc_routes, "retry_cc_statement_upload", cc_helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/{clicked_id}/retry",
                data={"password": "goodpw"},  # no save_password
            )

    assert resp.status_code == 303
    cc_helper.assert_awaited_once_with(clicked_id, "goodpw")


@pytest.mark.anyio
async def test_bank_retry_does_not_save_password_on_failure(session_factory):
    upload_id, account_id = await _seed_single_upload(
        session_factory, "bank", "password_required"
    )
    from httpx import ASGITransport, AsyncClient

    helper = AsyncMock(return_value=False)
    with patch.object(bank_routes, "retry_bank_statement_upload", helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/statements/bank/{upload_id}/retry",
                data={"password": "wrongpw", "save_password": "1"},
            )

    assert resp.status_code == 303
    helper.assert_awaited_once_with(upload_id, "wrongpw")
    assert await _get_account_password(session_factory, account_id) is None


# ---------------------------------------------------------------------------
# Statement date-range helpers (used to scope the db_txns query during retry)
# ---------------------------------------------------------------------------


class _FakeCcTxn:
    def __init__(self, date: str):
        self.date = date


class _FakeCcParsed:
    def __init__(self, txn_dates: list[str], payment_dates: list[str] | None = None):
        self.transactions = [_FakeCcTxn(d) for d in txn_dates]
        self.payments_refunds = [_FakeCcTxn(d) for d in (payment_dates or [])]


class _FakeBankTxn:
    def __init__(self, date: str):
        self.date = date


class _FakeBankParsed:
    def __init__(
        self,
        txn_dates: list[str],
        period_start: str | None = None,
        period_end: str | None = None,
    ):
        self.transactions = [_FakeBankTxn(d) for d in txn_dates]
        self.statement_period_start = period_start
        self.statement_period_end = period_end


def test_cc_date_range_spans_transactions_and_payments():
    import datetime as _dt

    parsed = _FakeCcParsed(
        txn_dates=["05/03/2026", "20/03/2026"],
        payment_dates=["25/02/2026", "15/03/2026"],
    )
    result = dates_module.cc_stmt_date_range(parsed)
    assert result is not None
    lo, hi = result
    assert lo == _dt.date(2026, 2, 25)
    assert hi == _dt.date(2026, 3, 20)


def test_cc_date_range_returns_none_when_no_parseable_dates():
    parsed = _FakeCcParsed(txn_dates=[], payment_dates=[])
    assert dates_module.cc_stmt_date_range(parsed) is None


def test_cc_date_range_skips_unparseable_entries():
    import datetime as _dt

    parsed = _FakeCcParsed(txn_dates=["10/03/2026", "not-a-date", "15/03/2026"])
    result = dates_module.cc_stmt_date_range(parsed)
    assert result is not None
    lo, hi = result
    assert lo == _dt.date(2026, 3, 10)
    assert hi == _dt.date(2026, 3, 15)


def test_bank_date_range_prefers_declared_period():
    import datetime as _dt

    parsed = _FakeBankParsed(
        txn_dates=["15/03/2026"],
        period_start="01/03/2026",
        period_end="31/03/2026",
    )
    result = dates_module.bank_stmt_date_range(parsed)
    assert result is not None
    lo, hi = result
    assert lo == _dt.date(2026, 3, 1)
    assert hi == _dt.date(2026, 3, 31)


def test_bank_date_range_falls_back_to_transaction_dates():
    import datetime as _dt

    parsed = _FakeBankParsed(txn_dates=["10/03/2026", "25/03/2026"])
    result = dates_module.bank_stmt_date_range(parsed)
    assert result is not None
    lo, hi = result
    assert lo == _dt.date(2026, 3, 10)
    assert hi == _dt.date(2026, 3, 25)
