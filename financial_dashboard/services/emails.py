"""Email processing helpers."""

import datetime
import logging
from decimal import Decimal
from typing import NamedTuple
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError

from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Card,
    Email,
    EmailKind,
    StatementUpload,
    async_session,
)
from financial_dashboard.integrations.email.body import (
    _extract_html_body,
    _extract_text_body,
    _save_failed_email,
)
from financial_dashboard.integrations.email.parsing import (
    _extract_message_metadata,
    _parse_email_date,
)
from bank_email_parser import parse_email
from bank_email_parser.exceptions import ParseError, UnsupportedEmailTypeError
from bank_email_parser.models import ParsedEmail
from financial_dashboard.services.cc_disambiguation import (
    resolve_cc_payment_account,
    should_auto_reconcile_statement,
)
from financial_dashboard.services.linker import link_transaction
from financial_dashboard.services.reminders import check_payment_received
from financial_dashboard.services.settings import (
    get_setting_int,
    get_telegram_chat_id,
)
from financial_dashboard.services.statements.bank import (
    BankStatementProcessingError,
    process_bank_statement_email,
)
from financial_dashboard.services.statements.cc import (
    process_cc_statement_email_summary,
    process_statement_email,
)
from financial_dashboard.services.telegram import (
    build_account_label,
    send_bulk_summary,
    send_disambiguation_prompt,
    send_enrichment_notification,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)


def _serialize_datetime(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _is_duplicate_transaction_error(exc: IntegrityError) -> bool:
    message = str(exc.orig)
    # Accept both the stale name (uq_transaction_dedup, still in the message
    # check for backwards compat) and the real one (uq_transactions_ref per
    # db/models.py:259).
    return (
        "uq_transactions_ref" in message
        or "uq_transaction_dedup" in message
        or ("UNIQUE constraint failed:" in message and "transactions." in message)
    )


_IST = ZoneInfo("Asia/Kolkata")

# Email types known to emit transaction time in 12-hour format with no
# AM/PM marker. Defined in services/parser_quirks because the same set
# also gates the SMS-side alias-merge in services/txn_merge.
from financial_dashboard.services.parser_quirks import (  # noqa: E402
    AMBIGUOUS_12H_TIME_EMAIL_TYPES as _AMBIGUOUS_12H_TIME_EMAIL_TYPES,
)

# A few minutes of slack to absorb clock skew between the bank and the
# Date header. Beyond this, a candidate that lands *after* received_at
# is treated as in-the-future and rejected.
_RECEIVED_AT_FUTURE_TOLERANCE = datetime.timedelta(minutes=5)

# Bank transaction alerts arrive seconds-to-minutes after the
# transaction. A candidate more than this far before received_at is
# implausible and rejected — including the next-day-midnight case where
# the body date is yesterday and both candidates are >12h in the past,
# at which point the data is unrecoverable from the email alone and we
# leave the parsed time as-is.
_RECEIVED_AT_PAST_LIMIT = datetime.timedelta(hours=12)


def _disambiguate_am_pm(
    parsed_time: datetime.time,
    transaction_date: datetime.date | None,
    received_at: datetime.datetime | None,
) -> datetime.time:
    """Resolve AM/PM for a 12-hour parsed time by anchoring to the email's
    Date header.

    ICICI CC transaction-alert emails render time on a 12-hour clock and
    strip the AM/PM marker:
        '06:37:31' could be 06:37 AM or 06:37 PM
        '12:55:20' could be 12:55 AM (00:55) or 12:55 PM (12:55)

    Strategy: enumerate the two 24-hour candidates, anchor each to
    ``transaction_date`` in IST, reject candidates that are >5 min in
    the email's future (transactions can't post-date their own alert)
    or >12 h in the past (alerts arrive promptly), and pick the
    surviving candidate closest to ``received_at``.

    If ``received_at`` is missing or no candidate survives, the parsed
    time is returned unchanged — the data is unrecoverable from the
    email alone.
    """
    if transaction_date is None or received_at is None:
        return parsed_time
    hour = parsed_time.hour
    if hour == 0:
        # A 12-hour clock with stripped AM/PM produces hour ∈ {1..12};
        # hour 0 can only come from a 24-hour body and is already
        # unambiguous (midnight). Pass through untouched.
        return parsed_time
    if hour < 12:
        candidates = (parsed_time, parsed_time.replace(hour=hour + 12))
    elif hour == 12:
        # 12 in a 12-hour clock means 00 (midnight) OR 12 (noon).
        candidates = (parsed_time.replace(hour=0), parsed_time)
    else:
        # Body already on a 24-hour clock (hour > 12); unambiguous.
        return parsed_time

    if received_at.tzinfo is None:
        # Defensive: _parse_email_date returns aware datetimes, but if a
        # caller hands us a naive one, treat it as IST wall-time.
        received_ist = received_at.replace(tzinfo=_IST)
    else:
        received_ist = received_at.astimezone(_IST)

    best: tuple[datetime.timedelta, datetime.time] | None = None
    for cand_time in candidates:
        cand_dt = datetime.datetime.combine(transaction_date, cand_time, tzinfo=_IST)
        delta = cand_dt - received_ist
        # Reject candidates too far after the email arrived — the
        # transaction can't be in the email's future.
        if delta > _RECEIVED_AT_FUTURE_TOLERANCE:
            continue
        # Reject candidates implausibly far before the email arrived
        # — bank alerts don't lag the transaction by half a day.
        if -delta > _RECEIVED_AT_PAST_LIMIT:
            continue
        gap = abs(delta)
        if best is None or gap < best[0]:
            best = (gap, cand_time)
    return best[1] if best is not None else parsed_time


class ProcessedEmailParse(NamedTuple):
    error: str | None
    txn_data: dict | None
    password_hint: str | None
    parsed_email: ParsedEmail | None


def _process_email_full(bank: str, raw_bytes: bytes) -> ProcessedEmailParse:
    """Parse raw email bytes.

    Returns a ``ProcessedEmailParse`` NamedTuple of (error, txn_data,
    password_hint, parsed_email). ``parsed_email`` is the raw ``ParsedEmail``
    (or None if parsing failed), so callers can read ``parsed.statement``
    for summary-only emails. Positional unpacking is still supported.
    """
    html = _extract_html_body(raw_bytes)
    if not html:
        html = _extract_text_body(raw_bytes)
    if not html:
        return ProcessedEmailParse(
            "No HTML or text body found in email", None, None, None
        )

    try:
        parsed = parse_email(bank, html)
    except (ParseError, UnsupportedEmailTypeError) as e:
        return ProcessedEmailParse(str(e), None, None, None)

    password_hint = parsed.password_hint

    if (txn := parsed.transaction) is None:
        return ProcessedEmailParse(None, None, password_hint, parsed)

    transaction_date = txn.transaction_date
    received_at = _parse_email_date(raw_bytes)
    if transaction_date is None and received_at is not None:
        transaction_date = received_at.date()

    transaction_time = txn.transaction_time
    if (
        transaction_time is not None
        and parsed.email_type in _AMBIGUOUS_12H_TIME_EMAIL_TYPES
    ):
        transaction_time = _disambiguate_am_pm(
            transaction_time, transaction_date, received_at
        )

    return ProcessedEmailParse(
        None,
        {
            "bank": parsed.bank,
            "email_type": parsed.email_type,
            "direction": txn.direction,
            "amount": Decimal(str(txn.amount.amount)),
            "currency": txn.amount.currency,
            "transaction_date": transaction_date,
            "transaction_time": transaction_time,
            "counterparty": txn.counterparty,
            "card_mask": txn.card_mask,
            "account_mask": txn.account_mask,
            "reference_number": txn.reference_number,
            "channel": txn.channel,
            "balance": Decimal(str(txn.balance.amount)) if txn.balance else None,
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
    # CAS emails short-circuit: the bank/CC HTML parser doesn't apply, and we
    # don't want statement-fallback routing to run on them.
    if email_kind == EmailKind.CAS_STATEMENT:
        from financial_dashboard.services.cas_emails import process_cas_email

        cas_result: dict | None = None
        cas_error: str | None = None
        async with async_session() as cas_session:
            try:
                cas_result, cas_error = await process_cas_email(
                    cas_session, raw_bytes, source_id=source_id, log_ref=log_ref
                )
            except Exception as exc:
                # ingest_cas_payload does delete-then-insert; a mid-flow
                # failure that escaped process_cas_email would otherwise commit
                # the deletes with no replacement. Match bank/CC behaviour:
                # log + return an error, don't crash the poll cycle.
                logger.exception("CAS dispatch crashed for %s", log_ref)
                await cas_session.rollback()
                cas_error = f"unexpected {type(exc).__name__}: {exc}"
            else:
                if cas_result is not None:
                    await cas_session.commit()
                else:
                    # Caught error inside process_cas_email — same rollback story.
                    await cas_session.rollback()
        return EmailDispatchResult(cas_error, None, None, cas_result)

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

        bank_stmt_error: str | None = None
        if stmt_result is None and try_bank:
            try:
                stmt_result = await process_bank_statement_email(
                    bank,
                    raw_bytes,
                    subject,
                    source_id=source_id,
                    password_hint=password_hint,
                )
            except BankStatementProcessingError as stmt_err:
                bank_stmt_error = str(stmt_err)
                logger.warning(
                    "Bank statement processing error for %s: %s",
                    log_ref,
                    bank_stmt_error,
                )
            except Exception as stmt_err:
                bank_stmt_error = f"Unexpected {type(stmt_err).__name__}: {stmt_err}"
                logger.exception("Bank statement processing crashed for %s", log_ref)

        if stmt_result is None:
            logger.info(
                "Statement processing returned None for %s (no PDF or subject mismatch)",
                log_ref,
            )
            if is_statement_rule:
                error = bank_stmt_error or "Statement processing returned no result"

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
        if stmt_result.get("bank_statement_upload_id"):
            stmt_type = "bank"
        elif stmt_result.get("cas_upload_id"):
            stmt_type = "CAS"
        else:
            stmt_type = "CC"
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
    pending_enrichment_notifications: list[tuple[int, object, dict]] = []
    pending_payment_checks: list[tuple[int, int, object]] = []
    pending_disambiguations: list[dict] = []

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
            elif stmt_result and stmt_result.get("cas_upload_id"):
                from financial_dashboard.services.cas_emails import (
                    link_cas_upload_email,
                )

                await link_cas_upload_email(
                    session, stmt_result["cas_upload_id"], email_row.id
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
                from financial_dashboard.services.txn_merge import merge_transaction

                try:
                    outcome, txn_row, diff = await merge_transaction(
                        session, "email", txn_data, email_id=email_row.id
                    )
                except IntegrityError as exc:
                    # Defense-in-depth: merge_transaction already catches
                    # uq_transactions_ref races, so this branch should be
                    # unreachable. Keep it to preserve the existing behavior
                    # for any other unique constraint.
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
                    if outcome == "created":
                        link_transaction(link_context, txn_row)
                        await session.flush()
                        # Maskless CC bill-payment: try amount-match against
                        # open statements; else queue a Telegram prompt.
                        prompt_payload = await resolve_cc_payment_account(
                            session, txn_row
                        )
                        if prompt_payload is not None:
                            pending_disambiguations.append(prompt_payload)
                        # Queue statement reconciliation for true CC
                        # bill-payment credits only. Scoped to `created`:
                        # an enriched row (second source for an already-
                        # seen payment) must NOT re-fire — payment_paid_amount
                        # is cumulative and would double-count.
                        # The redundant `account_id is not None` check is
                        # for ty's benefit: should_auto_reconcile_statement
                        # already guarantees it at runtime, but ty can't
                        # narrow through helper calls.
                        if (
                            should_auto_reconcile_statement(txn_row)
                            and txn_row.account_id is not None
                        ):
                            pending_payment_checks.append(
                                (txn_row.id, txn_row.account_id, txn_row.amount)
                            )
                    elif outcome == "enriched":
                        # Re-link if the enrichment filled card_mask/account_mask
                        # and the row is still unlinked.
                        if txn_row.account_id is None and (
                            diff.filled.get("card_mask")
                            or diff.filled.get("account_mask")
                        ):
                            link_transaction(link_context, txn_row)
                            await session.flush()
                        # Queue an enrichment notification when this email
                        # added or overwrote fields on an existing row. The
                        # bulk dispatcher below drops these when one poll
                        # produces more than `telegram.bulk_threshold`
                        # primaries — they'd be lost in the summary anyway.
                        if should_notify and diff.changed_fields:
                            enrich_account = (
                                await session.get(Account, txn_row.account_id)
                                if txn_row.account_id
                                else None
                            )
                            enrich_card = (
                                await session.get(Card, txn_row.card_id)
                                if txn_row.card_id
                                else None
                            )
                            pending_enrichment_notifications.append(
                                (
                                    txn_row.id,
                                    diff,
                                    {
                                        "bank": txn_row.bank,
                                        "direction": txn_row.direction,
                                        "amount": txn_row.amount,
                                        "counterparty": txn_row.counterparty,
                                        "transaction_date": txn_row.transaction_date,
                                        "transaction_time": txn_row.transaction_time,
                                        "card_mask": txn_row.card_mask,
                                        "account_label": build_account_label(
                                            enrich_account, enrich_card
                                        ),
                                        "channel": txn_row.channel,
                                    },
                                )
                            )
                    if should_notify and outcome == "created":
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
            elif error:
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

    if pending_notifications:
        chat_id = get_telegram_chat_id()
        bulk_threshold = get_setting_int("telegram.bulk_threshold", 5)
        if len(pending_notifications) <= bulk_threshold:
            for txn_id, txn_info in pending_notifications:
                await send_transaction_notification(
                    txn_id, txn_info, chat_id, source="email"
                )
            # Enrichment notifications fire alongside primaries only on the
            # per-row dispatch path. The bulk-summary path below collapses
            # everything into one message and can't represent diffs.
            if pending_enrichment_notifications:
                for (
                    enrich_txn_id,
                    diff,
                    enrich_info,
                ) in pending_enrichment_notifications:
                    await send_enrichment_notification(
                        enrich_txn_id,
                        diff,
                        chat_id,
                        source="email",
                        txn_info=enrich_info,
                    )
        else:
            await send_bulk_summary(
                len(pending_notifications),
                chat_id,
                source="email",
                txns=pending_notifications,
            )
            # Drop per-row enrichments when the batch took the bulk path.
            if pending_enrichment_notifications:
                logger.info(
                    "Dropped %d enrichment notifications in bulk email poll",
                    len(pending_enrichment_notifications),
                )
    elif pending_enrichment_notifications:
        # No primaries fired (everything was enrichment-only); still emit
        # the per-row enrichments — they're not part of the bulk path.
        chat_id = get_telegram_chat_id()
        for enrich_txn_id, diff, enrich_info in pending_enrichment_notifications:
            await send_enrichment_notification(
                enrich_txn_id,
                diff,
                chat_id,
                source="email",
                txn_info=enrich_info,
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

    if pending_disambiguations:
        chat_id = get_telegram_chat_id()
        for payload in pending_disambiguations:
            try:
                await send_disambiguation_prompt(payload, chat_id)
            except Exception as exc:
                logger.warning(
                    "CC disambiguation prompt failed for txn %s: %s",
                    payload.get("txn_id"),
                    exc,
                )
