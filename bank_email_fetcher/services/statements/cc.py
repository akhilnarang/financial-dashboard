# ty: ignore
"""CC statement PDF parsing and reconciliation for bank-email-fetcher.

Provides:
- parse_statement(): parses a CC statement PDF file using cc-parser's
  extract_raw_pdf + get_parser("auto", ...) auto-detection pipeline.

- reconcile_statement(): matches statement transactions against existing DB
  transactions for an account by (date, amount, direction) with a ±1-day
  tolerance. Returns matched, missing (in statement but not DB), and extra
  (in DB but not statement) lists.

- enrich_matched_transactions(): writes statement narration back to the DB
  counterparty field for matched transactions where the existing counterparty
  is null or a generic placeholder (e.g. "payment received").

- extract_pdf_from_email(): extracts PDF attachments from raw RFC822 email
  bytes. Skips known non-statement PDFs (MITC, T&C docs).

- process_statement_email(): end-to-end pipeline called by fetcher.poll_all()
  when a normal email parse fails. Checks that the subject contains "statement",
  extracts the PDF, tries parsing with and without stored passwords, finds the
  matching Account, reconciles, auto-imports missing transactions, and creates
  a StatementUpload row.

- _find_account(): finds an existing credit card Account matching the
  statement's card last-4 (checking both account_number and cards table).
  Returns None when no match exists — statements do not auto-create accounts.

Inline imports (from bank_email_fetcher.db, bank_email_fetcher.linker) are
used inside async functions to avoid circular import issues.
"""

import asyncio
import datetime
import email as email_lib
import json
import logging
import re
import tempfile
from datetime import date as date_type, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from sqlalchemy import func, select
from bank_email_fetcher.db import (
    Account,
    Card,
    StatementUpload,
    Transaction,
    async_session,
)

from bank_email_fetcher.config import get_fernet
from bank_email_fetcher.core.dates import parse_date
from bank_email_fetcher.integrations.parsers import (
    format_cc_amount,
    parse_cc_statement_pdf,
    parse_cc_token_amount,
)
from bank_email_fetcher.services.linker import build_link_context, link_transaction
from bank_email_fetcher.services.statements.hint import extract_password_hint
from bank_email_fetcher.services.settings import (
    get_setting_int,
    get_telegram_chat_id,
    should_notify_transactions,
)
from bank_email_fetcher.services.telegram import (
    build_account_label,
    send_bulk_summary,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"


def parse_statement(pdf_path: Path, password: str | None = None, bank: str = "auto"):
    """Parse a CC statement PDF. Returns a ParsedStatement."""
    return parse_cc_statement_pdf(pdf_path, password, bank=bank)


def parse_cc_amount(amount_str: str) -> Decimal:
    """Convert cc-parser amount string '25,000.00' to Decimal."""
    return Decimal(amount_str.replace(",", ""))


def parse_cc_date(date_str: str) -> date_type:
    """Convert cc-parser date 'DD/MM/YYYY' to date object."""
    parsed = parse_date(date_str, dayfirst=True)
    if parsed is None:
        raise ValueError(f"Could not parse CC statement date: {date_str!r}")
    return parsed


def last4_from_card(card_str: str | None) -> str | None:
    """Extract last 4 digits from a card number string."""
    if not card_str:
        return None
    digits = re.sub(r"[^0-9]", "", card_str)
    return digits[-4:] if len(digits) >= 4 else None


def _extract_digits(card_str: str | None) -> str:
    """Extract all digit characters from a card string (even if < 4)."""
    if not card_str:
        return ""
    return re.sub(r"[^0-9]", "", card_str)


def _match_key(txn_date: date_type, amount: Decimal, direction: str) -> tuple:
    return (txn_date, amount, direction)


def reconcile_statement(parsed, db_transactions: list, account_id: int) -> dict:
    """Match statement transactions against DB transactions.

    Returns a dict with matched, missing, and extra lists.
    """
    # Build all statement transactions (debits + credits)
    # Adjustment pairs reference the same Transaction objects already in
    # parsed.transactions / parsed.payments_refunds, so we do NOT re-add them.
    stmt_txns = []
    for txn in parsed.transactions or []:
        stmt_txns.append(("transactions", "debit", txn))
    for txn in parsed.payments_refunds or []:
        stmt_txns.append(("payments_refunds", "credit", txn))

    # Build DB candidate pool indexed by (date, amount, direction) for fast lookup
    # Each key maps to a list of DB transactions (multiple txns can share the same key)
    db_pool: dict[tuple, list] = {}
    for db_txn in db_transactions:
        if db_txn.transaction_date and db_txn.amount is not None:
            key = _match_key(
                db_txn.transaction_date, Decimal(str(db_txn.amount)), db_txn.direction
            )
            db_pool.setdefault(key, []).append(db_txn)

    matched = []
    missing = []

    for stmt_idx, (stmt_list, direction, txn) in enumerate(stmt_txns):
        try:
            amount = parse_cc_amount(txn.amount)
            txn_date = parse_cc_date(txn.date)
        except ValueError, InvalidOperation:
            # Can't parse — treat as missing
            missing.append(
                {
                    "stmt_idx": stmt_idx,
                    "stmt_list": stmt_list,
                    "date": txn.date,
                    "amount": txn.amount,
                    "direction": direction,
                    "narration": txn.narration,
                    "card_number": txn.card_number,
                    "person": txn.person,
                    "imported": False,
                    "imported_txn_id": None,
                }
            )
            continue

        # Try exact date, then +/-1 day
        found = False
        for date_offset in (0, -1, 1):
            candidate_date = txn_date + timedelta(days=date_offset)
            key = _match_key(candidate_date, amount, direction)
            candidates = db_pool.get(key, [])
            if candidates:
                db_txn = candidates.pop(0)  # greedy: take first match
                if not candidates:
                    del db_pool[key]
                matched.append(
                    {
                        "stmt_idx": stmt_idx,
                        "stmt_list": stmt_list,
                        "date": txn.date,
                        "amount": txn.amount,
                        "direction": direction,
                        "narration": txn.narration,
                        "card_number": txn.card_number,
                        "person": txn.person,
                        "db_txn_id": db_txn.id,
                        "db_counterparty": db_txn.counterparty,
                        "db_reference": db_txn.reference_number,
                        "db_date": str(db_txn.transaction_date),
                    }
                )
                found = True
                break

        if not found:
            missing.append(
                {
                    "stmt_idx": stmt_idx,
                    "stmt_list": stmt_list,
                    "date": txn.date,
                    "amount": txn.amount,
                    "direction": direction,
                    "narration": txn.narration,
                    "card_number": txn.card_number,
                    "person": txn.person,
                    "imported": False,
                    "imported_txn_id": None,
                }
            )

    return {
        "matched": matched,
        "missing": missing,
        "card_summaries": [
            {
                "card_number": cs.card_number,
                "person": cs.person,
                "transaction_count": cs.transaction_count,
                "total_amount": cs.total_amount,
                "reward_points_total": cs.reward_points_total,
            }
            for cs in (parsed.card_summaries or [])
        ],
        "payments_refunds_total": parsed.payments_refunds_total,
        "adjustment_pairs": [
            {
                "pair_id": p.pair_id,
                "kind": p.kind,
                "confidence": p.confidence,
                "score": p.score,
                "debit_narration": p.debit.narration if p.debit else None,
                "debit_amount": p.debit.amount if p.debit else None,
                "debit_date": p.debit.date if p.debit else None,
                "credit_narration": p.credit.narration if p.credit else None,
                "credit_amount": p.credit.amount if p.credit else None,
                "credit_date": p.credit.date if p.credit else None,
                "amount_delta": p.amount_delta,
            }
            for p in (parsed.possible_adjustment_pairs or [])
        ],
        "adjustments_debit_total": _calculate_adjustment_total(
            parsed.possible_adjustment_pairs or [], "debit"
        ),
        "adjustments_credit_total": _calculate_adjustment_total(
            parsed.possible_adjustment_pairs or [], "credit"
        ),
        "overall_total": parsed.overall_total,
        "overall_reward_points": parsed.overall_reward_points,
    }


def _calculate_adjustment_total(pairs, direction: str) -> str:
    """Calculate total adjustment amount for high-confidence pairs in given direction."""
    from decimal import Decimal

    total = Decimal("0")
    for pair in pairs:
        if pair.confidence == "high":
            if direction == "debit" and pair.debit:
                total += parse_cc_token_amount(pair.debit.amount or "0")
            elif direction == "credit" and pair.credit:
                total += parse_cc_token_amount(pair.credit.amount or "0")

    return format_cc_amount(total)


_GENERIC_COUNTERPARTIES = {"payment received", "payment successful", "payment done"}


async def enrich_matched_transactions(recon: dict) -> int:
    """Update DB transaction counterparty from statement narration for matched transactions.

    Enriches when the DB counterparty is NULL, empty, or a generic placeholder.
    Returns the count of transactions that were updated.
    """
    enriched = 0
    async with async_session() as session:
        for entry in recon.get("matched", []):
            narration = (entry.get("narration") or "").strip()
            if not narration:
                continue

            db_txn_id = entry.get("db_txn_id")
            if not db_txn_id:
                continue

            txn = await session.get(Transaction, db_txn_id)
            if not txn:
                continue

            existing = (txn.counterparty or "").strip()
            if existing and existing.lower() not in _GENERIC_COUNTERPARTIES:
                continue  # already has a meaningful counterparty

            txn.counterparty = narration
            enriched += 1
            entry["enriched"] = True

        if enriched:
            await session.commit()

    return enriched


def reconciliation_to_json(data: dict) -> str:
    """Serialize reconciliation data to JSON."""
    return json.dumps(data)


def reconciliation_from_json(data: str) -> dict:
    """Deserialize reconciliation data from JSON."""
    return json.loads(data)


def group_recon_by_person(recon: dict) -> list[dict]:
    """Group debit-transaction reconciliation entries by person for per-card display.

    Only groups entries with stmt_list=="transactions" (debits). Payments/refunds
    and adjustments are shown in their own global sections.

    Returns a list of person groups with matched/imported entries and summary info.
    Returns [] if only 1 unique person (signals flat layout).
    """
    from collections import defaultdict

    card_summaries = recon.get("card_summaries", [])
    cs_by_person = {(cs["person"] or "Unknown"): cs for cs in card_summaries}

    # Only group debit transactions — payments/refunds/adjustments have their own sections
    all_entries: list[tuple[str, dict]] = []
    for entry in recon.get("matched", []):
        if entry.get("stmt_list") == "transactions":
            all_entries.append(("matched", entry))
    for entry in recon.get("missing", []):
        if entry.get("imported") and entry.get("stmt_list") == "transactions":
            all_entries.append(("imported", entry))

    persons: set[str] = set()
    for _, entry in all_entries:
        persons.add(entry.get("person") or "")

    if len(persons) <= 1:
        return []

    groups: dict[str, dict] = defaultdict(
        lambda: {
            "matched": [],
            "imported": [],
            "card_numbers": set(),
        }
    )
    for entry_type, entry in all_entries:
        person = entry.get("person") or "Unknown"
        groups[person][entry_type].append(entry)
        cn = entry.get("card_number")
        if cn:
            groups[person]["card_numbers"].add(cn)

    result = []
    for person in sorted(groups.keys()):
        g = groups[person]
        card_numbers = sorted(g["card_numbers"])
        summary = cs_by_person.get(person)
        result.append(
            {
                "person": person,
                "card_number": card_numbers[0] if card_numbers else None,
                "matched": g["matched"],
                "imported": g["imported"],
                "matched_count": len(g["matched"]),
                "imported_count": len(g["imported"]),
                "total_count": len(g["matched"]) + len(g["imported"]),
                "summary": summary,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Email-based statement processing
# ---------------------------------------------------------------------------

_SKIP_PDF_NAMES = {
    "most important terms",
    "mitc",
    "terms & conditions",
    "terms and conditions",
    "tnc",
}


def extract_pdf_from_email(raw_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extract PDF attachments from raw RFC822 email bytes."""
    msg = email_lib.message_from_bytes(raw_bytes)
    pdfs = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            filename = part.get_filename() or ""
            # Match application/pdf OR octet-stream with .pdf filename
            is_pdf = ct == "application/pdf" or (
                ct == "application/octet-stream" and filename.lower().endswith(".pdf")
            )
            if not is_pdf:
                continue
            # Skip known non-statement PDFs (MITC, T&C docs)
            if any(skip in filename.lower() for skip in _SKIP_PDF_NAMES):
                logger.debug("Skipping non-statement PDF: %s", filename)
                continue
            pdf_bytes = part.get_payload(decode=True)
            if pdf_bytes:
                pdfs.append((filename or "statement.pdf", pdf_bytes))
    return pdfs


def _parse_pdf_bytes_sync(
    pdf_bytes: bytes, password: str | None = None, bank: str = "auto"
):
    """Save PDF bytes to temp file, parse, and clean up. Returns ParsedStatement."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = Path(f.name)
    try:
        return parse_statement(tmp_path, password, bank=bank)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _find_account(bank: str, parsed) -> "Account | None":
    """Find an existing credit_card account matching the statement's card.

    Returns None if nothing matches — statements must not auto-create accounts.
    """
    stmt_card_last4 = last4_from_card(parsed.card_number)
    # Some banks (e.g. SBI) only show 2 digits: "XXXX XXXX XXXX XX67"
    stmt_partial = _extract_digits(parsed.card_number) if not stmt_card_last4 else ""

    async with async_session() as session:
        cc_accounts = (
            (
                await session.execute(
                    select(Account).where(
                        func.lower(Account.bank) == bank.lower(),
                        Account.type == "credit_card",
                        Account.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )

    # Try to match by last-4 of card number
    account = None
    if stmt_card_last4:
        # Check account numbers
        for acc in cc_accounts:
            if last4_from_card(acc.account_number) == stmt_card_last4:
                account = acc
                break
        # Check cards table
        if not account:
            async with async_session() as session:
                cards = (await session.execute(select(Card))).scalars().all()
                for card in cards:
                    if last4_from_card(card.card_mask) == stmt_card_last4:
                        for acc in cc_accounts:
                            if acc.id == card.account_id:
                                account = acc
                                break
                        if account:
                            break
    elif stmt_partial:
        # Suffix match: bank only provides partial digits (e.g. "67" from SBI)
        for acc in cc_accounts:
            acc_l4 = last4_from_card(acc.account_number)
            if acc_l4 and acc_l4.endswith(stmt_partial):
                account = acc
                break
        if not account:
            async with async_session() as session:
                cards = (await session.execute(select(Card))).scalars().all()
                for card in cards:
                    card_l4 = last4_from_card(card.card_mask)
                    if card_l4 and card_l4.endswith(stmt_partial):
                        for acc in cc_accounts:
                            if acc.id == card.account_id:
                                account = acc
                                break
                        if account:
                            break

    if account:
        return account

    logger.info(
        "No matching credit_card account for bank=%s card=%s; statement not imported",
        bank,
        parsed.card_number,
    )
    return None


async def process_statement_email(
    bank: str,
    raw_bytes: bytes,
    email_subject: str,
    source_id: int | None = None,
    password_hint: str | None = None,
) -> dict | None:
    """Try to process an email as a CC statement.

    Returns a dict with statement_upload_id and stats if successful, None otherwise.
    """
    # Only process emails whose subject indicates a CC statement
    subject_lower = (email_subject or "").lower()
    if "statement" not in subject_lower:
        logger.debug(
            "Skipping non-statement email: %r",
            email_subject[:80] if email_subject else "",
        )
        return None
    # Reject bank/savings statements — CC subjects mention "card".
    if (
        "account statement" in subject_lower
        or "bank statement" in subject_lower
        or "savings statement" in subject_lower
    ) and "card" not in subject_lower:
        logger.debug(
            "Skipping bank account statement (not CC): %r",
            email_subject[:80] if email_subject else "",
        )
        return None

    # Require at least one CC account for this bank; statements must not
    # auto-create accounts.
    async with async_session() as session:
        has_cc_account = (
            await session.execute(
                select(Account.id).where(
                    func.lower(Account.bank) == bank.lower(),
                    Account.type == "credit_card",
                    Account.active.is_(True),
                )
            )
        ).first() is not None
    if not has_cc_account:
        logger.info(
            "Skipping CC statement path: no credit_card account for bank=%s", bank
        )
        return None

    if not password_hint:
        password_hint = extract_password_hint(raw_bytes, bank=bank)
    if password_hint:
        logger.info("Password hint: %s", password_hint)

    # Extract PDF attachments
    pdfs = extract_pdf_from_email(raw_bytes)
    if not pdfs:
        logger.info(
            "Statement email has no PDF attachment: bank=%s subject=%r",
            bank,
            email_subject[:80] if email_subject else "",
        )
        return None

    filename, pdf_bytes = pdfs[0]
    logger.info(
        "Found PDF attachment in statement email: bank=%s file=%s (%d bytes)",
        bank,
        filename,
        len(pdf_bytes),
    )

    # Parse the PDF — try without password first, then with stored passwords
    parsed = None
    try:
        parsed = await asyncio.to_thread(_parse_pdf_bytes_sync, pdf_bytes, None, bank)
    except ValueError as e:
        if "encrypt" not in str(e).lower() and "password" not in str(e).lower():
            logger.warning("Failed to parse statement PDF from email: %s", e)
            return None

        # PDF is encrypted — try stored passwords from credit card accounts
        # Use case-insensitive bank name matching so 'axis' (fetch rule)
        # matches 'Axis' (account)
        fernet = get_fernet()
        async with async_session() as session:
            cc_accounts = (
                (
                    await session.execute(
                        select(Account).where(
                            func.lower(Account.bank) == bank.lower(),
                            Account.type == "credit_card",
                            Account.active.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )

        passwords_to_try = []
        for acc in cc_accounts:
            if acc.statement_password:
                try:
                    pw = fernet.decrypt(acc.statement_password.encode()).decode()
                    passwords_to_try.append((acc, pw))
                except Exception:
                    pass

        for acc, pw in passwords_to_try:
            try:
                parsed = await asyncio.to_thread(_parse_pdf_bytes_sync, pdf_bytes, pw, bank)
                logger.info(
                    "Decrypted statement PDF using stored password for %s (%s)",
                    bank,
                    acc.label,
                )
                break
            except Exception:
                continue

        if not parsed:
            # No stored password worked — save for manual retry.
            STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            safe_name = filename.replace("/", "_").replace("\\", "_")
            file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
            file_path.write_bytes(pdf_bytes)

            # Only attach if there's exactly one CC account for this bank;
            # otherwise we'd mis-attribute the PDF.
            account = cc_accounts[0] if len(cc_accounts) == 1 else None
            if not account:
                logger.warning(
                    "Encrypted CC statement received but %d candidate accounts for bank=%s — leaving unassigned (%s)",
                    len(cc_accounts),
                    bank,
                    safe_name,
                )
                return None

            async with async_session() as session:
                upload = StatementUpload(
                    account_id=account.id,
                    bank=account.bank,
                    filename=safe_name,
                    file_path=str(file_path),
                    status="password_required",
                    error="PDF is encrypted — provide password via Statements page",
                )
                if (
                    password_hint
                    and not account.statement_password_hint
                    and (account_row := await session.get(Account, account.id))
                ):
                    account_row.statement_password_hint = password_hint
                session.add(upload)
                await session.commit()
                logger.info(
                    "Encrypted CC statement saved for manual password entry: %s",
                    safe_name,
                )
                return {
                    "statement_upload_id": upload.id,
                    "matched": 0,
                    "missing": 0,
                    "imported": 0,
                }
    except Exception as e:
        logger.warning("Failed to parse statement PDF from email: %s", e)
        return None

    account = await _find_account(bank, parsed)
    if account is None:
        return None

    # Reconcile
    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account.id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_statement(parsed, db_txns, account.id)

    # Save the PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace("/", "_").replace("\\", "_")
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    file_path.write_bytes(pdf_bytes)

    # Create StatementUpload and import missing transactions
    async with async_session() as session:
        upload = StatementUpload(
            account_id=account.id,
            bank=parsed.bank or bank,
            filename=safe_name,
            file_path=str(file_path),
            status="parsed",
            card_number=parsed.card_number,
            statement_name=parsed.name,
            due_date=parsed.due_date,
            total_amount_due=parsed.statement_total_amount_due,
            parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
            matched_count=len(recon["matched"]),
            missing_count=len(recon["missing"]),
            reconciliation_data=reconciliation_to_json(recon),
        )
        session.add(upload)
        await session.flush()

        # Auto-import all missing transactions
        link_ctx = await build_link_context(session)

        # Build a lookup to resolve partial card digits (e.g. SBI "67" → "0567")
        # against cards registered for this account.
        acct_cards = (
            (await session.execute(select(Card).where(Card.account_id == account.id)))
            .scalars()
            .all()
        )
        _card_l4s = [last4_from_card(c.card_mask) for c in acct_cards]
        _card_l4s = [v for v in _card_l4s if v]

        def _resolve_card_mask(raw: str | None) -> str | None:
            l4 = last4_from_card(raw)
            if l4:
                return l4
            partial = _extract_digits(raw)
            if partial:
                for cl4 in _card_l4s:
                    if cl4.endswith(partial):
                        return cl4
            return last4_from_card(account.account_number)

        imported = 0
        imported_txns: list[tuple[int, dict]] = []
        for entry in recon["missing"]:
            try:
                amount = parse_cc_amount(entry["amount"])
                txn_date = parse_cc_date(entry["date"])
            except ValueError, KeyError:
                continue

            resolved_mask = _resolve_card_mask(entry.get("card_number"))

            txn = Transaction(
                statement_upload_id=upload.id,
                account_id=account.id,
                bank=bank,
                email_type="cc_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("narration"),
                card_mask=resolved_mask,
                channel="cc_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()

            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1
            account_obj = (
                await session.get(Account, txn.account_id) if txn.account_id else None
            )
            card_obj = await session.get(Card, txn.card_id) if txn.card_id else None
            imported_txns.append(
                (
                    txn.id,
                    {
                        "bank": txn.bank,
                        "direction": txn.direction,
                        "amount": txn.amount,
                        "counterparty": txn.counterparty,
                        "transaction_date": txn.transaction_date,
                        "transaction_time": txn.transaction_time,
                        "card_mask": txn.card_mask,
                        "account_label": build_account_label(account_obj, card_obj),
                        "channel": txn.channel,
                    },
                )
            )

        upload.imported_count = imported
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"  # all matched or all imported
        elif imported > 0:
            upload.status = "partial_import"
        await session.commit()

        if imported_txns and should_notify_transactions():
            chat_id = get_telegram_chat_id()
            bulk_threshold = get_setting_int("telegram.bulk_threshold", 5)
            if len(imported_txns) <= bulk_threshold:
                for txn_id, txn_info in imported_txns:
                    await send_transaction_notification(txn_id, txn_info, chat_id)
            else:
                await send_bulk_summary(
                    len(imported_txns),
                    chat_id,
                    account_label=account.label,
                    source="cc_statement",
                    txns=imported_txns,
                )

        enriched = await enrich_matched_transactions(recon)

        # function-local: breaks cycle with services.reminders (reminders imports services.statements at top)
        from bank_email_fetcher.services.reminders import init_payment_tracking

        await init_payment_tracking(upload.id)

        logger.info(
            "Processed statement email: bank=%s account=%s matched=%d missing=%d imported=%d enriched=%d",
            bank,
            account.label,
            len(recon["matched"]),
            len(recon["missing"]),
            imported,
            enriched,
        )
        return {
            "statement_upload_id": upload.id,
            "matched": len(recon["matched"]),
            "missing": len(recon["missing"]),
            "imported": imported,
            "enriched": enriched,
        }
