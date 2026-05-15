"""Dashboard regression tests for credit-card outstanding grouping."""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import Request

from financial_dashboard.db import Account, Base, StatementUpload
from financial_dashboard.web import dashboard as dashboard_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


class CapturingTemplates:
    def __init__(self):
        self.context: dict[str, Any] | None = None

    def TemplateResponse(self, request, template_name, context):
        self.context = context
        return SimpleNamespace(
            request=request, template_name=template_name, context=context
        )


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace()),
        }
    )


def _captured_context(templates: CapturingTemplates) -> dict[str, Any]:
    assert templates.context is not None
    return templates.context


@pytest.mark.anyio
async def test_zero_due_statements_without_paid_status_are_grouped_as_paid(
    session_factory, monkeypatch
):
    templates = CapturingTemplates()
    monkeypatch.setattr(dashboard_module, "templates", templates)

    async with session_factory() as session:
        no_due_account = Account(
            bank="hdfc",
            label="No Due CC",
            type="credit_card",
            active=True,
        )
        no_payment_required_account = Account(
            bank="sbi",
            label="No Payment Required CC",
            type="credit_card",
            active=True,
        )
        session.add_all([no_due_account, no_payment_required_account])
        await session.flush()
        session.add_all(
            [
                StatementUpload(
                    account_id=no_due_account.id,
                    bank="hdfc",
                    filename="",
                    file_path="",
                    status="parsed",
                    due_date=None,
                    total_amount_due="0.00",
                    payment_status=None,
                ),
                StatementUpload(
                    account_id=no_payment_required_account.id,
                    bank="sbi",
                    filename="",
                    file_path="",
                    status="parsed",
                    due_date="NO PAYMENT REQUIRED",
                    total_amount_due="0.00",
                    payment_status="unpaid",
                ),
            ]
        )
        await session.commit()

        await dashboard_module.dashboard(_request(), session)

    cc_outstanding = _captured_context(templates)["cc_outstanding"]
    assert cc_outstanding["outstanding_rows"] == []
    assert [row["account"].label for row in cc_outstanding["paid_rows"]] == [
        "No Due CC",
        "No Payment Required CC",
    ]
    assert cc_outstanding["summary"]["cards_paid"] == 2
    assert cc_outstanding["summary"]["cards_with_outstanding"] == 0


@pytest.mark.anyio
async def test_outstanding_rows_link_to_latest_statement(session_factory, monkeypatch):
    templates = CapturingTemplates()
    monkeypatch.setattr(dashboard_module, "templates", templates)

    async with session_factory() as session:
        account = Account(
            bank="hdfc",
            label="Primary CC",
            type="credit_card",
            active=True,
        )
        session.add(account)
        await session.flush()
        upload = StatementUpload(
            account_id=account.id,
            bank="hdfc",
            filename="",
            file_path="",
            status="parsed",
            due_date=None,
            total_amount_due="1234.56",
            payment_status="unpaid",
        )
        session.add(upload)
        await session.commit()

        await dashboard_module.dashboard(_request(), session)

    row = _captured_context(templates)["cc_outstanding"]["outstanding_rows"][0]
    assert row["statement_url"] == f"/statements/{upload.id}"
