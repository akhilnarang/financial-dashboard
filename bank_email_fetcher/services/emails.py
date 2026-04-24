"""Email processing helpers."""

from __future__ import annotations

import datetime
import logging

from sqlalchemy.exc import IntegrityError

from bank_email_fetcher.db import (
    Account,
    BankStatementUpload,
    Card,
    Email,
    EmailKind,
    StatementUpload,
    Transaction,
    async_session,
)
from bank_email_fetcher.integrations.email.body import (
    _extract_html_body,
    _extract_text_body,
    _save_failed_email,
)
from bank_email_fetcher.integrations.email.parsing import (
    _extract_message_metadata,
    _parse_email_date,
)
from bank_email_parser import parse_email
from bank_email_parser.exceptions import ParseError, UnsupportedEmailTypeError
from bank_email_parser.models import ParsedEmail
from bank_email_fetcher.services.linker import link_transaction
from bank_email_fetcher.services.reminders import check_payment_received
from bank_email_fetcher.services.settings import (
    get_setting_int,
    get_telegram_chat_id,
)
from bank_email_fetcher.services.statements.bank import process_bank_statement_email
from bank_email_fetcher.services.statements.cc import process_statement_email
from bank_email_fetcher.services.telegram import (
    build_account_label,
    send_bulk_summary,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)


def _serialize_datetime(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _is_duplicate_transaction_error(exc: IntegrityError) -> bool:
    message = str(exc.orig)
    return "uq_transaction_dedup" in message or (
        "UNIQUE constraint failed:" in message and "transactions." in message
    )


def _process_email(
    bank: str, raw_bytes: bytes
) -> tuple[str | None, dict | None, str | None]:
    """Parse raw email bytes. Returns (error, txn_dict, password_hint)."""
    html = _extract_html_body(raw_bytes)
    if not html:
        html = _extract_text_body(raw_bytes)
    if not html:
        return "No HTML or text body found in email", None, None

    try:
        parsed = parse_email(bank, html)
    except (ParseError, UnsupportedEmailTypeError) as e:
        return str(e), None, None

    if (txn := parsed.transaction) is None:
        return None, None, parsed.password_hint

    transaction_date = txn.transaction_date
    if transaction_date is None:
        received_at = _parse_email_date(raw_bytes)
        if received_at is not None:
            transaction_date = received_at.date()

    return (
        None,
        {
            "bank": parsed.bank,
            "email_type": parsed.email_type,
            "direction": txn.direction,
            "amount": float(txn.amount.amount),
            "currency": txn.amount.currency,
            "transaction_date": transaction_date,
            "transaction_time": txn.transaction_time,
            "counterparty": txn.counterparty,
            "card_mask": txn.card_mask,
            "account_mask": txn.account_mask,
            "reference_number": txn.reference_number,
            "channel": txn.channel,
            "balance": float(txn.balance.amount) if txn.balance else None,
            "raw_description": txn.raw_description,
        },
        None,
    )


_STATEMENT_KINDS = {
    EmailKind.CC_STATEMENT,
    EmailKind.BANK_STATEMENT,
    EmailKind.STATEMENT,
}


async def parse_email_by_kind(
    *,
    bank: str,
    email_kind: str | None,
    raw_bytes: bytes,
    subject: str,
    source_id: int | None,
    log_ref: str,
) -> tuple[str | None, dict | None, str | None, dict | None]:
    """Run txn and/or statement pipelines based on the rule's email_kind.

    Returns ``(error, txn_data, password_hint, stmt_result)`` — exactly one of
    txn_data or stmt_result will be populated on success.
    """
    error: str | None = None
    txn_data: dict | None = None
    password_hint: str | None = None
    stmt_result: dict | None = None

    if email_kind not in _STATEMENT_KINDS:
        error, txn_data, password_hint = _process_email(bank, raw_bytes)

    try_cc = email_kind in (EmailKind.CC_STATEMENT, EmailKind.STATEMENT) or (
        email_kind in (None, EmailKind.TRANSACTION) and not txn_data
    )
    try_bank = email_kind in (EmailKind.BANK_STATEMENT, EmailKind.STATEMENT) or (
        email_kind in (None, EmailKind.TRANSACTION) and not txn_data
    )

    if try_cc or try_bank:
        logger.info(
            "Email %s %s (bank=%s, kind=%s, subject=%r), trying statement path",
            log_ref,
            "routed to statement pipeline"
            if email_kind in _STATEMENT_KINDS
            else "failed parsing",
            bank,
            email_kind,
            subject[:80],
        )
        if try_cc:
            try:
                stmt_result = await process_statement_email(
                    bank,
                    raw_bytes,
                    subject,
                    source_id=source_id,
                    password_hint=password_hint,
                )
            except Exception as stmt_err:
                logger.warning(
                    "CC statement processing error for %s: %s", log_ref, stmt_err
                )

        if stmt_result is None and try_bank:
            try:
                stmt_result = await process_bank_statement_email(
                    bank,
                    raw_bytes,
                    subject,
                    source_id=source_id,
                    password_hint=password_hint,
                )
            except Exception as stmt_err:
                logger.warning(
                    "Bank statement processing error for %s: %s", log_ref, stmt_err
                )

        if stmt_result is None:
            logger.info(
                "Statement processing returned None for %s (no PDF or subject mismatch)",
                log_ref,
            )
            if email_kind in _STATEMENT_KINDS:
                error = "Statement processing returned no result"

    return error, txn_data, password_hint, stmt_result


async def handle_polled_email(
    *,
    rule,
    provider: str,
    source_id: int,
    msg_id: str,
    remote_id: str,
    raw_bytes: bytes,
    should_notify: bool,
    link_context,
    stats: dict,
) -> None:
    metadata = _extract_message_metadata(raw_bytes)
    received_at = _parse_email_date(raw_bytes)

    email_kind = getattr(rule, "email_kind", None)
    subject = metadata.get("subject", "")
    error, txn_data, password_hint, stmt_result = await parse_email_by_kind(
        bank=rule.bank,
        email_kind=email_kind,
        raw_bytes=raw_bytes,
        subject=subject,
        source_id=source_id,
        log_ref=msg_id,
    )

    if stmt_result:
        error = None
        stats["parsed"] += 1
        stmt_type = "bank" if stmt_result.get("bank_statement_upload_id") else "CC"
        logger.info(
            "Processed %s statement from email %s: matched=%d imported=%d",
            stmt_type,
            msg_id,
            stmt_result["matched"],
            stmt_result["imported"],
        )
    elif error:
        try:
            _save_failed_email(provider, msg_id, raw_bytes)
        except Exception as save_err:
            logger.warning("Could not save failed email to spool: %s", save_err)

    pending_notifications: list[tuple[int, dict]] = []
    pending_payment_checks: list[tuple[int, int, object]] = []

    async with async_session() as session:
        async with session.begin():
            if stmt_result:
                initial_status = "parsed"
            else:
                initial_status = (
                    "pending" if txn_data else ("failed" if error else "skipped")
                )
            email_row = Email(
                provider=provider,
                message_id=msg_id,
                source_id=source_id,
                remote_id=remote_id,
                sender=metadata["sender"],
                subject=metadata["subject"],
                received_at=received_at,
                status=initial_status,
                error=error,
                rule_id=rule.id,
            )
            session.add(email_row)
            await session.flush()

            if stmt_result and stmt_result.get("statement_upload_id"):
                su_id = stmt_result["statement_upload_id"]
                su = await session.get(StatementUpload, su_id)
                if su:
                    su.email_id = email_row.id
                else:
                    logger.warning(
                        "StatementUpload %s disappeared before email %s could be linked",
                        su_id,
                        msg_id,
                    )
            elif stmt_result and stmt_result.get("bank_statement_upload_id"):
                su_id = stmt_result["bank_statement_upload_id"]
                su = await session.get(BankStatementUpload, su_id)
                if su:
                    su.email_id = email_row.id
                else:
                    logger.warning(
                        "BankStatementUpload %s disappeared before email %s could be linked",
                        su_id,
                        msg_id,
                    )

            skip_txn_types = {"sbi_cc_transaction_declined"}

            if txn_data and txn_data.get("email_type") in skip_txn_types:
                email_row.status = "parsed"
                email_row.error = None
                stats["parsed"] += 1
                if should_notify:
                    txn_data["_declined"] = True
                    pending_notifications.append((0, txn_data))
            elif txn_data:
                try:
                    async with session.begin_nested():
                        txn_row = Transaction(email_id=email_row.id, **txn_data)
                        session.add(txn_row)
                        await session.flush()
                except IntegrityError as exc:
                    if not _is_duplicate_transaction_error(exc):
                        raise
                    email_row.status = "skipped"
                    email_row.error = "Duplicate transaction skipped because an identical transaction row already exists"
                    stats["skipped"] += 1
                    logger.warning(
                        "Skipping duplicate transaction for email %s (rule=%s, source=%s): %s",
                        msg_id,
                        rule.id,
                        source_id,
                        exc.orig,
                    )
                else:
                    email_row.status = "parsed"
                    email_row.error = None
                    stats["parsed"] += 1
                    link_transaction(link_context, txn_row)
                    await session.flush()
                    if should_notify:
                        account_obj = (
                            await session.get(Account, txn_row.account_id)
                            if txn_row.account_id
                            else None
                        )
                        card_obj = (
                            await session.get(Card, txn_row.card_id)
                            if txn_row.card_id
                            else None
                        )
                        pending_notifications.append(
                            (
                                txn_row.id,
                                {
                                    "bank": txn_row.bank,
                                    "direction": txn_row.direction,
                                    "amount": txn_row.amount,
                                    "counterparty": txn_row.counterparty,
                                    "transaction_date": txn_row.transaction_date,
                                    "transaction_time": txn_row.transaction_time,
                                    "card_mask": txn_row.card_mask,
                                    "account_label": build_account_label(
                                        account_obj, card_obj
                                    ),
                                    "channel": txn_row.channel,
                                },
                            )
                        )
                    if txn_row.direction == "credit" and txn_row.account_id:
                        pending_payment_checks.append(
                            (txn_row.id, txn_row.account_id, txn_row.amount)
                        )
            elif error:
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

    if pending_notifications:
        chat_id = get_telegram_chat_id()
        bulk_threshold = get_setting_int("telegram.bulk_threshold", 5)
        if len(pending_notifications) <= bulk_threshold:
            for txn_id, txn_info in pending_notifications:
                await send_transaction_notification(txn_id, txn_info, chat_id)
        else:
            await send_bulk_summary(
                len(pending_notifications),
                chat_id,
                source="email",
                txns=pending_notifications,
            )

    if pending_payment_checks:
        for txn_id, acct_id, amt in pending_payment_checks:
            try:
                await check_payment_received(txn_id, acct_id, amt)
            except Exception as exc:
                logger.warning(
                    "Payment-received check failed for txn %s: %s",
                    txn_id,
                    exc,
                )
