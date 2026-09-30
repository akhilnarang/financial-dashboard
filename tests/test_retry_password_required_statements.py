"""Tests for the password-required retry coordinator and the manual retry
routes. The per-upload retry helpers are mocked."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.main import create_app
import financial_dashboard.core.deps as core_deps
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
)
from financial_dashboard.services import accounts as accounts_module
from financial_dashboard.services.statements.dates import cc_stmt_date_range
from financial_dashboard.web import bank_statements as bank_routes
from financial_dashboard.web import statements as cc_routes
from tests.conftest import new_test_engine


@pytest.fixture
async def session_factory(monkeypatch):
    """Swap request/session factories for an in-memory DB."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


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
async def test_coordinator_retries_password_required_only_and_counts_failures(
    session_factory, monkeypatch
):
    """Other statuses are skipped. A helper that returns False or raises
    counts as failed, and a raise does not abort the loop."""
    account_id = await _seed_account_with_uploads(
        session_factory,
        cc_statuses=["password_required", "password_required", "parsed"],
        bank_statuses=["password_required", "imported"],
    )

    cc_helper = AsyncMock(side_effect=[RuntimeError("boom"), True])
    bank_helper = AsyncMock(return_value=False)
    monkeypatch.setattr(accounts_module, "retry_cc_statement_upload", cc_helper)
    monkeypatch.setattr(accounts_module, "retry_bank_statement_upload", bank_helper)
    async with session_factory() as session:
        result = await accounts_module.retry_password_required_statements(
            session, account_id, "secret"
        )

    assert cc_helper.await_count == 2
    assert bank_helper.await_count == 1
    assert result == {
        "cc_retried": 1,
        "bank_retried": 0,
        "cc_failed": 1,
        "bank_failed": 1,
    }


# The manual retry endpoints save the password only when the retry succeeds
# and save_password=1. Saving it also retries the sibling uploads.


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


def _password_gated_helper(maker, model):
    """Succeed only for "goodpw" and flip the upload status like the real
    retry helper does."""

    async def _side_effect(upload_id, password):
        if password != "goodpw":
            return False
        async with maker() as session:
            if upload := await session.get(model, upload_id):
                upload.status = "parsed"
                await session.commit()
        return True

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.anyio
async def test_cc_retry_saves_password_and_unlocks_siblings_only_on_success(
    session_factory,
):
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

    cc_helper = _password_gated_helper(session_factory, StatementUpload)
    bank_helper = _password_gated_helper(session_factory, BankStatementUpload)
    with (
        patch.object(cc_routes, "retry_cc_statement_upload", cc_helper),
        patch.object(accounts_module, "retry_cc_statement_upload", cc_helper),
        patch.object(accounts_module, "retry_bank_statement_upload", bank_helper),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            url = f"/statements/{clicked_id}/retry"
            # A wrong password with save_password saves nothing.
            resp = await client.post(
                url, data={"password": "wrongpw", "save_password": "1"}
            )
            assert resp.status_code == 303
            assert await _get_account_password(session_factory, account_id) is None
            # A good password without save_password retries only the clicked one.
            await client.post(url, data={"password": "goodpw"})
            assert cc_helper.await_count == 2
            assert bank_helper.await_count == 0
            assert await _get_account_password(session_factory, account_id) is None
            # With save_password the password is saved and siblings retry.
            await client.post(url, data={"password": "goodpw", "save_password": "1"})

    assert cc_helper.await_count == 4
    assert bank_helper.await_count == 1
    assert await _get_account_password(session_factory, account_id) is not None


@pytest.mark.anyio
async def test_bank_retry_saves_password_only_on_success(session_factory):
    upload_id, account_id = await _seed_single_upload(
        session_factory, "bank", "password_required"
    )
    helper = _password_gated_helper(session_factory, BankStatementUpload)
    with patch.object(bank_routes, "retry_bank_statement_upload", helper):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            url = f"/statements/bank/{upload_id}/retry"
            resp = await client.post(
                url, data={"password": "wrongpw", "save_password": "1"}
            )
            assert resp.status_code == 303
            assert await _get_account_password(session_factory, account_id) is None
            await client.post(url, data={"password": "goodpw", "save_password": "1"})

    assert await _get_account_password(session_factory, account_id) is not None


def test_cc_date_range_spans_transactions_and_payments():
    parsed = SimpleNamespace(
        transactions=[
            SimpleNamespace(date=d) for d in ["05/03/2026", "not-a-date", "20/03/2026"]
        ],
        payments_refunds=[
            SimpleNamespace(date=d) for d in ["25/02/2026", "15/03/2026"]
        ],
    )
    assert cc_stmt_date_range(parsed) == (date(2026, 2, 25), date(2026, 3, 20))
