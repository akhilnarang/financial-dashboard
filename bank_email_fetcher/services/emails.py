"""Email processing helpers."""

from __future__ import annotations

import datetime
import logging
from typing import NamedTuple

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
from bank_email_fetcher.services.statements.cc import (
    process_cc_statement_email_summary,
    process_statement_email,
)
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


def _process_email_full(
    bank: str, raw_bytes: bytes
) -> tuple[str | None, dict | None, str | None, ParsedEmail | None]:
    """Parse raw email bytes.

    Returns ``(error, txn_dict, password_hint, parsed_email)`` — ``parsed_email``
    is the raw ``ParsedEmail`` (or None if parsing failed), so callers can read
    ``parsed.statement`` for summary-only emails.
    """
    html = _extract_html_body(raw_bytes)
    if not html:
        html = _extract_text_body(raw_bytes)
    if not html:
        return "No HTML or text body found in email", None, None, None

    try:
        parsed = parse_email(bank, html)
    except (ParseError, UnsupportedEmailTypeError) as e:
        return str(e), None, None, None

    password_hint = parsed.password_hint

    if (txn := parsed.transaction) is None:
        return None, None, password_hint, parsed

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
        password_hint,
        parsed,
    )


_STATEMENT_KINDS = {
    EmailKind.CC_STATEMENT,
    EmailKind.BANK_STATEMENT,
    EmailKind.STATEMENT,
}


class EmailDispatchResult(NamedTuple):
    """Outcome of dispatching one raw email to the parse+route pipeline.

    Tuple-compatible: existing callers that unpack positionally
    (``error, txn_data, password_hint, stmt_result = ...``) continue to work.
    """

    error: str | None
    txn_data: dict | None
    password_hint: str | None
    stmt_result: dict | None


async def _dispatch_email_summary(
    *,
    bank: str,
    parsed_email: ParsedEmail,
    password_hint: str | None,
    log_ref: str,
) -> EmailDispatchResult:
    """Dispatch an email whose body already carries a ``StatementSummary``.

    Caller has already established that routing is allowed for this kind and
    that ``parsed_email.statement`` is populated. The returned result is
    final — the PDF path is NOT attempted when the summary path was taken,
    even if the handler refused (see summary-precedence comment in caller).

    Args:
        bank: Bank identifier from the matching ``FetchRule``.
        parsed_email: The parser's output; ``.statement`` MUST be populated
            (caller-enforced).
        password_hint: Any password hint the HTML parser extracted; threaded
            through to the result unchanged. Summary emails don't need a
            password, but the hint may still be useful for a later PDF upload
            on the same account.
        log_ref: Identifier (usually ``msg_id``) for log correlation.

    Returns:
        An ``EmailDispatchResult`` with ``stmt_result`` set on success, or
        with ``error`` explaining why the summary handler refused (no
        matching CC account, ambiguous match, or incomplete payload).
    """
    stmt_result: dict | None = None
    handler_error: str | None = None
    try:
        # ``email_id=None``: the ``emails`` row does not exist yet here;
        # it's linked after ``parse_email_by_kind`` returns, in
        # ``handle_polled_email`` / the reparse route.
        stmt_result = await process_cc_statement_email_summary(
            bank, parsed_email, email_id=None
        )
    except Exception as stmt_err:
        logger.warning(
            "CC statement summary processing error for %s: %s", log_ref, stmt_err
        )
        handler_error = f"Statement summary processing error: {stmt_err}"

    # The parser emitted a statement summary — this email *is* a statement.
    # Distinguish handler exceptions from handler refusals so the email's
    # ``error`` column doesn't lie about what went wrong.
    if stmt_result is not None:
        error: str | None = None
    elif handler_error is not None:
        error = handler_error
    else:
        error = (
            "Statement summary could not be persisted "
            "(no matching CC account or ambiguous match — see logs)"
        )
    return EmailDispatchResult(error, None, password_hint, stmt_result)


async def parse_email_by_kind(
    *,
    bank: str,
    email_kind: str | None,
    raw_bytes: bytes,
    subject: str,
    source_id: int | None,
    log_ref: str,
) -> EmailDispatchResult:
    """Run txn and/or statement pipelines based on the rule's email_kind.

    Exactly one of ``txn_data`` or ``stmt_result`` is populated on success.
    """
    # Always run the HTML parser — statement-kind emails still carry a
    # ``password_hint`` and may carry a full ``StatementSummary`` in the body
    # that the statement pipeline needs. For statement routing we discard the
    # parse error (typically "no transaction parser matched" — expected for a
    # statement email) and only keep hint + summary.
    raw_error, txn_data, password_hint, parsed_email = _process_email_full(
        bank, raw_bytes
    )

    # Compute routing booleans once.
    is_statement_rule = email_kind in _STATEMENT_KINDS
    is_transaction_rule = email_kind == EmailKind.TRANSACTION
    is_untyped_rule = email_kind is None
    allow_summary_route = email_kind in (
        EmailKind.CC_STATEMENT,
        EmailKind.STATEMENT,
        None,
    )

    if is_statement_rule:
        # Statement rules don't surface a transaction even if the parser
        # happened to populate one (wrong routing / ambiguous email type).
        txn_data = None
        error: str | None = None
    else:
        error = raw_error

    # Email-summary statements take precedence over PDF parsing: if the body
    # already carries a ``StatementSummary``, we persist from that and never
    # attempt the PDF path, even if the handler refuses.
    if (
        allow_summary_route
        and parsed_email is not None
        and parsed_email.statement is not None
    ):
        return await _dispatch_email_summary(
            bank=bank,
            parsed_email=parsed_email,
            password_hint=password_hint,
            log_ref=log_ref,
        )

    stmt_result: dict | None = None
    # Txn-capable rules (None / TRANSACTION) only try statement pipelines
    # as a fallback when transaction parsing didn't produce anything.
    fallback_to_stmt = (is_untyped_rule or is_transaction_rule) and not txn_data
    try_cc = (
        email_kind in (EmailKind.CC_STATEMENT, EmailKind.STATEMENT) or fallback_to_stmt
    )
    try_bank = (
        email_kind in (EmailKind.BANK_STATEMENT, EmailKind.STATEMENT)
        or fallback_to_stmt
    )

    if try_cc or try_bank:
        logger.info(
            "Email %s %s (bank=%s, kind=%s, subject=%r), trying statement path",
            log_ref,
            "routed to statement pipeline" if is_statement_rule else "failed parsing",
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
            if is_statement_rule:
                error = "Statement processing returned no result"

    return EmailDispatchResult(error, txn_data, password_hint, stmt_result)


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

    email_kind = rule.email_kind
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
        if stmt_result.get("summary_only"):
            logger.info(
                "Processed %s statement summary (email-only) from %s",
                stmt_type,
                msg_id,
            )
        else:
            logger.info(
                "Processed %s statement from email %s: matched=%d imported=%d",
                stmt_type,
                msg_id,
                stmt_result.get("matched", 0),
                stmt_result.get("imported", 0),
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
