"""Statement retry helpers extracted from app routes."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from financial_dashboard.core.dates import parse_date
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
    async_session,
)
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.snapshots import emit_bank_snapshot, emit_cc_snapshot
from financial_dashboard.services.statements.dates import (
    bank_stmt_date_range,
    cc_stmt_date_range,
)
from financial_dashboard.services.statements.bank import (
    _last4 as bank_last4,
    _parse_amount as parse_bank_amount,
    enrich_matched_transactions as enrich_bank_matched_transactions,
    parse_bank_statement,
    reconcile_bank_statement,
)
from financial_dashboard.services.statements.cc import (
    enrich_matched_transactions,
    import_missing_cc_txns,
    parse_statement,
    reconcile_statement,
    reconciliation_to_json,
)

logger = logging.getLogger(__name__)

STMT_RECONCILE_DATE_BUFFER_DAYS = 7


async def retry_cc_statement_upload(
    upload_id: int,
    password: str,
) -> bool:
    async with async_session() as session:
        if not (upload := await session.get(StatementUpload, upload_id)):
            return False
        account_id = upload.account_id
        file_path = upload.file_path
        bank = upload.bank
        source_kind = upload.source_kind

    if source_kind == "email_summary":
        logger.info(
            "Skipping retry for CC statement #%s: email_summary has no PDF to reparse",
            upload_id,
        )
        return False

    if not file_path:
        logger.info(
            "Skipping retry for CC statement #%s: no file_path on disk",
            upload_id,
        )
        return False

    try:
        parsed = await asyncio.to_thread(
            parse_statement, Path(file_path), password, bank
        )
    except Exception as exc:
        async with async_session() as session:
            if upload := await session.get(StatementUpload, upload_id):
                upload.error = str(exc)
                await session.commit()
        return False

    async with async_session() as session:
        stmt = select(Transaction).where(Transaction.account_id == account_id)
        if date_range := cc_stmt_date_range(parsed):
            lo, hi = date_range
            stmt = stmt.where(
                Transaction.transaction_date.between(
                    lo - timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
                    hi + timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
                )
            )
        db_txns = list((await session.execute(stmt)).scalars().all())

        recon = reconcile_statement(parsed, db_txns, account_id)
        await enrich_matched_transactions(recon)

        upload = await session.get(StatementUpload, upload_id)
        account = await session.get(Account, account_id)
        if upload is None or account is None:
            # The upload was deleted (or its account was) between the
            # first transaction and this one. Bail; nothing safe to
            # reconcile against.
            return False
        upload.status = "parsed"
        upload.bank = parsed.bank
        upload.card_number = parsed.card_number
        upload.statement_name = parsed.name
        upload.due_date = parsed.due_date
        upload.total_amount_due = parsed.statement_total_amount_due
        upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
        upload.matched_count = len(recon["matched"])
        upload.missing_count = len(recon["missing"])
        upload.reconciliation_data = reconciliation_to_json(recon)
        upload.error = None

        imported = len(
            await import_missing_cc_txns(session, upload, parsed, account, recon)
        )

        upload.imported_count = imported
        upload.missing_count = sum(
            1 for entry in recon["missing"] if not entry.get("imported")
        )
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        elif imported > 0:
            upload.status = "partial_import"
        await emit_cc_snapshot(session, upload)
        await session.commit()

    # function-local: breaks cycle with services.reminders (reminders imports services.statements at top)
    from financial_dashboard.services.reminders import init_payment_tracking

    await init_payment_tracking(upload_id)
    return True


async def retry_bank_statement_upload(
    upload_id: int,
    password: str,
) -> bool:
    async with async_session() as session:
        if not (upload := await session.get(BankStatementUpload, upload_id)):
            return False
        account_id = upload.account_id
        file_path = upload.file_path
        bank = upload.bank

    try:
        parsed = await asyncio.to_thread(
            parse_bank_statement, Path(file_path), bank, password
        )
    except Exception as exc:
        message = str(exc)
        is_password_error = (
            "encrypt" in message.lower() or "password" in message.lower()
        )
        async with async_session() as session:
            if upload := await session.get(BankStatementUpload, upload_id):
                upload.error = message
                if not is_password_error:
                    upload.status = "parse_error"
                await session.commit()
        return False

    async with async_session() as session:
        stmt = select(Transaction).where(Transaction.account_id == account_id)
        if date_range := bank_stmt_date_range(parsed):
            lo, hi = date_range
            stmt = stmt.where(
                Transaction.transaction_date.between(
                    lo - timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
                    hi + timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
                )
            )
        db_txns = list((await session.execute(stmt)).scalars().all())

        recon = reconcile_bank_statement(parsed, db_txns, account_id)
        await enrich_bank_matched_transactions(recon)

        upload = await session.get(BankStatementUpload, upload_id)
        if upload is None:
            # The upload was deleted between the first txn and this one.
            return False
        upload.status = "parsed"
        upload.account_number = parsed.account_number
        upload.account_holder_name = parsed.account_holder_name
        upload.opening_balance = parsed.opening_balance
        upload.closing_balance = parsed.closing_balance
        upload.statement_period_start = parsed.statement_period_start
        upload.statement_period_end = parsed.statement_period_end
        upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
        upload.matched_count = len(recon["matched"])
        upload.missing_count = len(recon["missing"])
        upload.reconciliation_data = reconciliation_to_json(recon)
        upload.error = None

        link_ctx = await build_link_context(session)
        imported = 0
        for entry in recon["missing"]:
            if entry.get("imported"):
                continue
            try:
                amount = parse_bank_amount(entry["amount"])
                txn_date = parse_date(entry["date"], dayfirst=True)
            except KeyError, TypeError, ValueError:
                txn_date = None
            if txn_date is None:
                continue
            txn = Transaction(
                bank_statement_upload_id=upload_id,
                account_id=account_id,
                bank=parsed.bank,
                email_type="bank_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("counterparty") or entry.get("narration"),
                account_mask=bank_last4(parsed.account_number),
                reference_number=entry.get("reference_number"),
                channel=entry.get("channel") or "bank_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()
            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1

        upload.imported_count = imported
        upload.missing_count = sum(
            1 for entry in recon["missing"] if not entry.get("imported")
        )
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        elif imported > 0:
            upload.status = "partial_import"
        await emit_bank_snapshot(session, upload)
        await session.commit()

    return True


async def retry_password_required_statements(
    account_id: int,
    password: str,
    retry_cc_upload=retry_cc_statement_upload,
    retry_bank_upload=retry_bank_statement_upload,
) -> dict:
    result = {"cc_retried": 0, "bank_retried": 0, "cc_failed": 0, "bank_failed": 0}

    async with async_session() as session:
        cc_ids = (
            (
                await session.execute(
                    select(StatementUpload.id).where(
                        StatementUpload.account_id == account_id,
                        StatementUpload.status == "password_required",
                    )
                )
            )
            .scalars()
            .all()
        )
        bank_ids = (
            (
                await session.execute(
                    select(BankStatementUpload.id).where(
                        BankStatementUpload.account_id == account_id,
                        BankStatementUpload.status == "password_required",
                    )
                )
            )
            .scalars()
            .all()
        )

    for upload_id in cc_ids:
        try:
            ok = await retry_cc_upload(upload_id, password)
        except Exception as exc:
            ok = False
            logger.warning(
                "Auto-retry raised for CC statement %d on account %d: %s",
                upload_id,
                account_id,
                exc,
            )
        result["cc_retried" if ok else "cc_failed"] += 1

    for upload_id in bank_ids:
        try:
            ok = await retry_bank_upload(upload_id, password)
        except Exception as exc:
            ok = False
            logger.warning(
                "Auto-retry raised for bank statement %d on account %d: %s",
                upload_id,
                account_id,
                exc,
            )
        result["bank_retried" if ok else "bank_failed"] += 1

    return result
