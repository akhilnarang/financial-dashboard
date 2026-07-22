"""Regression coverage for reparsing recognized non-ledger emails."""

from decimal import Decimal
from email.message import EmailMessage
from unittest.mock import AsyncMock, patch

import pytest
from bank_email_parser.models import ParsedEmail
from sqlalchemy import select

from financial_dashboard.db import Email, FetchRule, Transaction
from financial_dashboard.integrations.email.body import RawEmailResult
from financial_dashboard.services.emails import (
    EmailDispatchResult,
    ProcessedEmailParse,
    parse_email_by_kind,
)

pytestmark = pytest.mark.anyio


def _usage_control_notice() -> bytes:
    message = EmailMessage()
    message["Subject"] = "Synthetic card settings notice"
    message["From"] = "notices@example.invalid"
    message.set_content(
        "We have applied the Usage Settings on your ICICI Bank Credit Card "
        "XX1234. Open Manage Your Cards, then Manage Credit Card Usage."
    )
    return message.as_bytes()


async def _seed_failed_emails(session, count: int) -> list[Email]:
    rule = FetchRule(
        provider="gmail",
        sender="notices@example.invalid",
        bank="icici",
        enabled=True,
        email_kind=None,
    )
    session.add(rule)
    await session.flush()

    emails = []
    for index in range(count):
        email_row = Email(
            provider="gmail",
            message_id=f"synthetic-notice-{index}@example.invalid",
            sender="notices@example.invalid",
            subject="Synthetic card settings notice",
            status="failed",
            error="Previous synthetic parser failure",
            rule_id=rule.id,
        )
        session.add(email_row)
        emails.append(email_row)
    await session.commit()
    return emails


async def test_recognized_notice_stays_non_transaction_for_statement_rule():
    with (
        patch(
            "financial_dashboard.services.emails.process_statement_email",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "financial_dashboard.services.emails.process_bank_statement_email",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await parse_email_by_kind(
            bank="icici",
            email_kind="cc_statement",
            raw_bytes=_usage_control_notice(),
            subject="Synthetic card settings notice",
            source_id=None,
            log_ref="synthetic-notice",
        )

    assert result.error is None
    assert result.txn_data is None
    assert result.stmt_result is None
    assert result.recognized_non_transaction is True


async def test_unprocessed_statement_remains_an_error():
    parsed_statement = ParsedEmail(
        bank="synthetic",
        email_type="synthetic_account_statement",
    )
    with (
        patch(
            "financial_dashboard.services.emails._process_email_full",
            return_value=ProcessedEmailParse(None, None, None, parsed_statement),
        ),
        patch(
            "financial_dashboard.services.emails.process_statement_email",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "financial_dashboard.services.emails.process_bank_statement_email",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await parse_email_by_kind(
            bank="synthetic",
            email_kind=None,
            raw_bytes=b"synthetic",
            subject="Synthetic statement",
            source_id=None,
            log_ref="synthetic-statement",
        )

    assert result.error == "Statement processing returned no result"
    assert result.txn_data is None
    assert result.stmt_result is None
    assert result.recognized_non_transaction is False


async def test_single_reparse_marks_real_non_transaction_parse_skipped(client, session):
    [email_row] = await _seed_failed_emails(session, 1)
    email_id = email_row.id
    raw_email = _usage_control_notice()

    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw_email, None, "provider")),
        ),
        patch(
            "financial_dashboard.services.emails.process_statement_email",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "financial_dashboard.services.emails.process_bank_statement_email",
            new=AsyncMock(return_value=None),
        ),
        patch("financial_dashboard.web.emails._save_failed_email") as save_failed,
    ):
        response = await client.post(f"/api/emails/{email_id}/reparse")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "message": "Email contained no ledger event and was skipped",
        "new_status": "skipped",
        "txn_id": None,
    }
    save_failed.assert_not_called()

    updated_email = await session.get(Email, email_id)
    assert updated_email is not None
    assert updated_email.status == "skipped"
    assert updated_email.error is None
    assert (await session.execute(select(Transaction))).scalars().all() == []


@pytest.mark.parametrize(
    ("existing_status", "attach_transaction"),
    [("parsed", False), ("failed", True)],
)
async def test_single_reparse_refuses_existing_ledger_state(
    client, session, existing_status, attach_transaction
):
    [email_row] = await _seed_failed_emails(session, 1)
    email_id = email_row.id
    email_row.status = existing_status
    expected_error = (
        None if existing_status == "parsed" else "Previous synthetic parser failure"
    )
    email_row.error = expected_error
    if attach_transaction:
        session.add(
            Transaction(
                email_id=email_id,
                bank="synthetic",
                email_type="synthetic_transaction",
                direction="debit",
                amount=Decimal("1.00"),
            )
        )
    await session.commit()

    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(
                return_value=RawEmailResult(_usage_control_notice(), None, "provider")
            ),
        ),
        patch(
            "financial_dashboard.web.emails.parse_email_by_kind",
            new=AsyncMock(
                return_value=EmailDispatchResult(None, None, None, None, True)
            ),
        ),
    ):
        response = await client.post(f"/api/emails/{email_id}/reparse")

    assert response.status_code == 409, response.text
    assert response.json() == {
        "detail": (
            "Only failed emails without existing ledger links can be marked as "
            "non-transaction notices"
        )
    }
    unchanged_email = await session.get(Email, email_id)
    assert unchanged_email is not None
    assert unchanged_email.status == existing_status
    assert unchanged_email.error == expected_error
    transaction_count = len(
        (await session.execute(select(Transaction))).scalars().all()
    )
    assert transaction_count == int(attach_transaction)


async def test_bulk_reparse_separates_non_transactions_from_parse_failures(
    client, session
):
    recognized_email, failed_email = await _seed_failed_emails(session, 2)
    recognized_email_id = recognized_email.id
    failed_email_id = failed_email.id
    raw_email = _usage_control_notice()

    async def parse_result_for_email(**kwargs):
        if kwargs["log_ref"] == f"bulk-reparse:{recognized_email_id}":
            return EmailDispatchResult(None, None, None, None, True)
        return EmailDispatchResult("Synthetic parse failure", None, None, None, False)

    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw_email, None, "provider")),
        ),
        patch(
            "financial_dashboard.web.emails.parse_email_by_kind",
            side_effect=parse_result_for_email,
        ),
        patch("financial_dashboard.web.emails._save_failed_email") as save_failed,
    ):
        response = await client.post("/emails/reparse-all-failed")

    assert response.status_code == 200, response.text
    assert response.json() == {"succeeded": 0, "skipped": 1, "failed": 1}
    assert save_failed.call_count == 1

    updated_recognized_email = await session.get(Email, recognized_email_id)
    updated_failed_email = await session.get(Email, failed_email_id)
    assert updated_recognized_email is not None
    assert updated_failed_email is not None
    assert updated_recognized_email.status == "skipped"
    assert updated_recognized_email.error is None
    assert updated_failed_email.status == "failed"
    assert updated_failed_email.error == "Previous synthetic parser failure"


async def test_non_transaction_reparse_does_not_overwrite_concurrent_change(
    client, session
):
    [email_row] = await _seed_failed_emails(session, 1)
    email_id = email_row.id
    raw_email = _usage_control_notice()

    async def load_after_concurrent_change(_email_row):
        current_email = await session.get(Email, email_id)
        assert current_email is not None
        current_email.status = "parsed"
        current_email.error = None
        await session.commit()
        return RawEmailResult(raw_email, None, "provider")

    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            side_effect=load_after_concurrent_change,
        ),
        patch(
            "financial_dashboard.web.emails.parse_email_by_kind",
            new=AsyncMock(
                return_value=EmailDispatchResult(None, None, None, None, True)
            ),
        ),
    ):
        response = await client.post(f"/api/emails/{email_id}/reparse")

    assert response.status_code == 409, response.text
    assert response.json() == {
        "detail": "Email changed while it was being reparsed; retry the request"
    }
    updated_email = await session.get(Email, email_id)
    assert updated_email is not None
    assert updated_email.status == "parsed"
    assert updated_email.error is None
