# ty: ignore
"""Bank account statement PDF parsing and reconciliation.

Parallel to statements.py (which handles CC statements). Provides:

- parse_bank_statement(): parses a bank account statement PDF using
  bank-statement-parser's extract_raw_pdf + get_parser(bank).

- reconcile_bank_statement(): matches statement transactions against
  existing DB transactions by (date, amount, direction) with ±1-day
  tolerance and optional reference_number matching.

- enrich_matched_transactions(): writes statement narration back to
  the DB counterparty field for matched transactions.

- process_bank_statement_email(): end-to-end pipeline called by the
  fetcher fallback chain when CC statement processing returns None.

Inline imports (from bank_email_fetcher.db, bank_email_fetcher.linker)
are used inside async functions to avoid circular import issues.
"""

import asyncio
import datetime
import json
import logging
import re
import tempfile
from datetime import date as date_type, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from bank_email_fetcher.db import (
    Account,
    BankStatementUpload,
    Card,
    Transaction,
    async_session,
)

from bank_email_fetcher.config import get_fernet
from bank_email_fetcher.core.dates import parse_date
from bank_email_fetcher.integrations.parsers import parse_bank_statement_pdf
from bank_email_fetcher.services.linker import build_link_context, link_transaction
from bank_email_fetcher.services.settings import (
    get_setting_int,
    get_telegram_chat_id,
    should_notify_transactions,
)
from bank_email_fetcher.services.statements.cc import extract_pdf_from_email
from bank_email_fetcher.services.statements.hint import extract_password_hint
from bank_email_fetcher.services.telegram import (
    build_account_label,
    send_bulk_summary,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"


class BankStatementProcessingError(Exception):
    """Raised when a bank statement was identified for processing but
    processing failed in a way the caller should surface to the user.

    Returning ``None`` from ``process_bank_statement_email`` is reserved
    for *skips* (not a bank statement, no PDF, no matching account, etc.)
    where the email simply doesn't apply. Anything past that boundary —
    PDF unparseable, encryption deadlock, post-parse import collapse —
    raises this error so the caller can put a real message in front of
    the user instead of "Statement processing returned no result".
    """


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(filename: str | None) -> str:
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_bank_statement(pdf_path: Path, bank: str, password: str | None = None):
    """Parse a bank account statement PDF. Returns a ParsedBankStatement."""
    return parse_bank_statement_pdf(pdf_path, bank, password)


def _parse_amount(amount_str: str) -> Decimal:
    """Convert amount string '25,000.00' to Decimal."""
    return Decimal(amount_str.replace(",", ""))


def _parse_date(date_str: str) -> date_type:
    """Convert 'DD/MM/YYYY' to date object."""
    parsed = parse_date(date_str, dayfirst=True)
    if parsed is None:
        raise ValueError(f"Could not parse bank statement date: {date_str!r}")
    return parsed


def _match_key(txn_date: date_type, amount: Decimal, direction: str) -> tuple:
    return (txn_date, amount, direction)


def _take_first_unconsumed(
    candidates: list[int] | None, consumed: set[int]
) -> int | None:
    """Return the first id in ``candidates`` not already consumed."""
    if not candidates:
        return None
    for cid in candidates:
        if cid not in consumed:
            return cid
    return None


# Tokens we never count as distinctive overlap evidence for fuzzy
# matching: banking nouns/verbs, transfer purpose words, generic header
# words. Compared upper-case after extracting alphabetic runs of length
# ≥ 4 from both narrations.
_OVERLAP_STOPWORDS: frozenset[str] = frozenset(
    {
        "PAYMENT", "PAYMENTS", "PAID",
        "CREDIT", "CREDITS", "CREDITED", "CRED",
        "DEBIT", "DEBITS", "DEBITED",
        "TRANSFER", "TRANSFERS", "TRANSFERRED", "SELFTRANSFER", "SELF",
        "BANK", "ACCOUNT", "BENEFICIARY", "REMITTER", "SENDER",
        "FROM", "VIA", "FOR", "WITH",
        "RECEIVED", "RECEIPT", "SENT",
        "AMOUNT", "BALANCE", "AVAILABLE", "INDIA", "INTERNATIONAL",
        "TRANSACTION", "TXNRECONCILE", "TXN",
        "REFUND", "REFUNDS", "REVERSAL", "REVERSED",
        "UPI", "IMPS", "NEFT", "RTGS", "ACH", "NACH",
        "SLICE", "ICICI", "HDFC", "AXIS", "KOTAK", "IDFC", "INDUSIND",
        "SBI", "YESBANK", "EQUITAS", "JUPITER", "ONECARD", "HSBC",
        "OKAXIS", "OKICICI", "OKHDFCBANK", "OKSBI", "YBL", "PAYTM",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z]{4,}")

# Minimum length for a reference number to count as substring evidence.
# Below this we risk matching on an accidental embedded run (a 4-digit
# number inside a longer txn id, a year fragment, etc.).
_MIN_REF_SUBSTRING_LEN = 6


def _ref_appears_in(ref: str | None, text: str | None) -> bool:
    """Word-boundary-aware containment check used as fuzzy match evidence.

    Plain ``ref in text`` is too permissive — a short ref can fall
    accidentally inside an unrelated longer numeric/alphanumeric run.
    We require:

    - ``ref`` is at least ``_MIN_REF_SUBSTRING_LEN`` chars (cuts out
      noise-prone short tokens), and
    - ``ref`` appears at an alphanumeric word boundary in ``text``
      (so e.g. ``"3456"`` does NOT match inside ``"12345678901234"``,
      but ``"123456789012"`` does match in ``"ref 123456789012 to"``).
    """
    if not ref or not text or len(ref) < _MIN_REF_SUBSTRING_LEN:
        return False
    return bool(
        re.search(
            r"(?<![A-Za-z0-9])" + re.escape(ref) + r"(?![A-Za-z0-9])",
            text,
        )
    )


def _significant_tokens(text: str | None, exclude: frozenset[str] | set[str]) -> set[str]:
    """Extract alphabetic tokens of length ≥ 4 (upper-cased), minus the
    excluded set. Used to test whether two narrations share a distinctive
    counterparty token."""
    if not text:
        return set()
    return {tok.upper() for tok in _TOKEN_RE.findall(text)} - exclude


def _holder_name_tokens(name: str | None) -> set[str]:
    """Tokens making up the account holder's own name. We exclude these
    when scoring overlap because they appear in BOTH self-transfer
    narrations and as the user's beneficiary name on inward transfers,
    so they are not distinctive evidence."""
    if not name:
        return set()
    return {tok.upper() for tok in _TOKEN_RE.findall(name)}


def _is_compatible(
    cand,
    *,
    stmt_ref: str | None,
    stmt_narration: str | None,
    stmt_channel: str | None,
    holder_tokens: set[str],
    is_fuzzy_date: bool,
) -> bool:
    """Decide whether a date+amount+direction-matching DB candidate could
    be the same logical transaction as a statement row.

    Compatibility rules:

    - Refs agree, or at least one side has no ref → compatible. This is
      the common case: email parsers often miss the ref, statement
      parsers usually capture one. Allowed at any date offset (the ±1
      day window stays for timezone slop in the no-ref-disagreement
      case).

    - Both rows have refs and they differ → ambiguous by ref alone.
      We *only* attempt this rescue at exact date — once both sides have
      refs, a mismatch is already a strong negative signal and we should
      not double the collision window on top. So we refuse outright when
      the date is fuzzy. At exact date we look for narration evidence:

        a) DB ref appears at a word boundary inside stmt narration (UPI
           case where the email-extracted UTR is embedded in the stmt
           narration but the stmt ref column carries the bank's internal
           txn id), OR
        b) Stmt ref appears at a word boundary inside DB raw_description
           (symmetric but rarer), OR
        c) Channel is UPI on BOTH sides AND counterparty narration
           shares at least one distinctive token (excluding the account
           holder's own name and a stopword list of banking/purpose
           words).

      Strict-channel-UPI on (c) is deliberate: IMPS self-transfers
      collide easily on the holder's name and lack distinctive merchant
      tokens, so we prefer false-split there over false-merge. The
      word-boundary check on (a)/(b) prevents short refs from matching
      accidentally as embedded substrings of unrelated long ids.

    - Otherwise → incompatible.
    """
    cand_ref = getattr(cand, "reference_number", None)
    cand_raw = getattr(cand, "raw_description", None) or getattr(
        cand, "counterparty", None
    )
    cand_channel = getattr(cand, "channel", None)

    if not (stmt_ref and cand_ref) or stmt_ref == cand_ref:
        return True

    if is_fuzzy_date:
        return False

    if _ref_appears_in(cand_ref, stmt_narration):
        return True
    if _ref_appears_in(stmt_ref, cand_raw):
        return True

    if (
        stmt_channel == "upi"
        and cand_channel == "upi"
        and (
            _significant_tokens(cand_raw, _OVERLAP_STOPWORDS | holder_tokens)
            & _significant_tokens(stmt_narration, _OVERLAP_STOPWORDS | holder_tokens)
        )
    ):
        return True

    return False


def _select_unique_compatible(
    candidates: list[int] | None,
    consumed: set[int],
    db_by_id: dict[int, object],
    *,
    stmt_ref: str | None,
    stmt_narration: str | None,
    stmt_channel: str | None,
    holder_tokens: set[str],
    is_fuzzy_date: bool,
) -> int | None:
    """Return the unique compatible candidate, or ``None``.

    Uniqueness is enforced in two tiers so distinctive narration evidence
    can break ties:

    1. If at least one candidate matches by ref-substring (DB ref ⊆ stmt
       narration, or stmt ref ⊆ DB raw_description, both word-bounded),
       restrict to that set. This is high-confidence evidence.
    2. Otherwise consider all compatible candidates.

    Within the chosen tier, return the candidate iff exactly one passes —
    never silently pick the first of multiple. Refusal pushes the row
    into ``missing`` so the operator can resolve manually rather than
    have us merge into the wrong sibling row.
    """
    if not candidates:
        return None

    pool = [cid for cid in candidates if cid not in consumed]
    if not pool:
        return None

    strong: list[int] = []
    compatible: list[int] = []
    for cid in pool:
        cand = db_by_id.get(cid)
        if cand is None:
            continue
        if not _is_compatible(
            cand,
            stmt_ref=stmt_ref,
            stmt_narration=stmt_narration,
            stmt_channel=stmt_channel,
            holder_tokens=holder_tokens,
            is_fuzzy_date=is_fuzzy_date,
        ):
            continue
        compatible.append(cid)
        cand_ref = getattr(cand, "reference_number", None)
        cand_raw = getattr(cand, "raw_description", None) or getattr(
            cand, "counterparty", None
        )
        if _ref_appears_in(cand_ref, stmt_narration) or _ref_appears_in(
            stmt_ref, cand_raw
        ):
            strong.append(cid)

    pool_to_use = strong or compatible
    if len(pool_to_use) == 1:
        return pool_to_use[0]
    return None


def _missing_entry(stmt_idx: int, direction: str, txn) -> dict:
    return {
        "stmt_idx": stmt_idx,
        "date": txn.date,
        "amount": txn.amount,
        "direction": direction,
        "narration": txn.narration,
        "counterparty": txn.counterparty,
        "reference_number": txn.reference_number,
        "channel": txn.channel,
        "balance": txn.balance,
        "imported": False,
    }


def _matched_entry(stmt_idx: int, direction: str, txn, found) -> dict:
    return {
        "stmt_idx": stmt_idx,
        "date": txn.date,
        "amount": txn.amount,
        "direction": direction,
        "narration": txn.narration,
        "counterparty": txn.counterparty,
        "reference_number": txn.reference_number,
        "channel": txn.channel,
        "balance": txn.balance,
        "db_txn_id": found.id,
        "db_counterparty": found.counterparty,
        "db_reference": found.reference_number,
        "db_date": str(found.transaction_date) if found.transaction_date else None,
    }


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile_bank_statement(parsed, db_transactions: list, account_id: int) -> dict:
    """Match statement transactions against DB transactions.

    Two-pass matching:

    1. Walk every statement row that has a ``reference_number`` and claim
       the DB row sharing ``(reference_number, direction)``. These are
       deterministic high-confidence matches.

    2. For statement rows still unmatched, fall back to
       ``(date ±1 day, amount, direction)``. The fallback is *reference-
       aware*: if the statement row carries a non-null ref and the DB
       candidate carries a different non-null ref, we refuse the fuzzy
       match — they cannot be the same logical transaction.

    Splitting these passes prevents an early statement row whose ref
    happens to be absent from the DB from greedily eating, via the
    date-fallback, a DB row that a later statement row legitimately owns
    by reference.

    Returns a dict with matched, missing lists and balance verification.
    """
    stmt_txns: list[tuple[str, object]] = [
        (txn.transaction_type, txn) for txn in parsed.transactions or []
    ]

    # Pre-parse amount/date once so every pass uses the same normalized form
    # and parse failures land in `missing` immediately.
    parsed_rows: list[tuple[int, str, object, Decimal | None, date_type | None]] = []
    matched = []
    missing = []
    for stmt_idx, (direction, txn) in enumerate(stmt_txns):
        try:
            amount = _parse_amount(txn.amount)
            txn_date = _parse_date(txn.date)
        except ValueError, InvalidOperation:
            missing.append(_missing_entry(stmt_idx, direction, txn))
            continue
        parsed_rows.append((stmt_idx, direction, txn, amount, txn_date))

    # Build DB candidate pools.
    # ref_pool: (reference_number, direction) — UPI refunds may reuse the
    # same ref with the opposite direction, so direction stays in the key.
    # date_pool: (date, amount, direction) for fuzzy fallback.
    db_by_id: dict[int, object] = {db_txn.id: db_txn for db_txn in db_transactions}
    ref_pool: dict[tuple[str, str], list[int]] = {}
    date_pool: dict[tuple, list[int]] = {}
    for db_txn in db_transactions:
        if db_txn.reference_number and db_txn.direction:
            ref_pool.setdefault((db_txn.reference_number, db_txn.direction), []).append(
                db_txn.id
            )
        if db_txn.transaction_date and db_txn.amount is not None and db_txn.direction:
            key = _match_key(
                db_txn.transaction_date,
                Decimal(str(db_txn.amount)),
                db_txn.direction,
            )
            date_pool.setdefault(key, []).append(db_txn.id)

    consumed: set[int] = set()
    matched_db_ids: dict[int, object] = {}

    # Pass 1: reference-number matches across all stmt rows.
    for stmt_idx, direction, txn, _amount, _txn_date in parsed_rows:
        if not txn.reference_number:
            continue
        ref_key = (txn.reference_number, direction)
        candidate_id = _take_first_unconsumed(ref_pool.get(ref_key), consumed)
        if candidate_id is None:
            continue
        consumed.add(candidate_id)
        matched_db_ids[stmt_idx] = db_by_id[candidate_id]

    # Pass 2: date+amount+direction fallback for stmt rows still unmatched.
    # Compatibility is ref-and-narration-aware (see ``_is_compatible``).
    # The ±1 day window stays as the date-pool walk so timezone-slop
    # cases (DB row dated one off, stmt is exact, neither side has a
    # disagreeing ref) still match. Per-candidate exactness for the
    # both-refs-disagree path lives inside ``_is_compatible`` via the
    # ``is_fuzzy_date`` flag.
    holder_tokens = _holder_name_tokens(parsed.account_holder_name)
    for stmt_idx, direction, txn, amount, txn_date in parsed_rows:
        if stmt_idx in matched_db_ids:
            continue
        stmt_ref = txn.reference_number
        for offset in (0, -1, 1):
            key = _match_key(txn_date + timedelta(days=offset), amount, direction)
            candidate_id = _select_unique_compatible(
                date_pool.get(key),
                consumed,
                db_by_id,
                stmt_ref=stmt_ref,
                stmt_narration=txn.narration,
                stmt_channel=txn.channel,
                holder_tokens=holder_tokens,
                is_fuzzy_date=offset != 0,
            )
            if candidate_id is None:
                continue
            consumed.add(candidate_id)
            matched_db_ids[stmt_idx] = db_by_id[candidate_id]
            break

    # Build matched / missing entries in original statement order.
    for stmt_idx, direction, txn, _amount, _txn_date in parsed_rows:
        cand = matched_db_ids.get(stmt_idx)
        if cand is not None:
            matched.append(_matched_entry(stmt_idx, direction, txn, cand))
        else:
            missing.append(_missing_entry(stmt_idx, direction, txn))

    # Balance verification
    balance_verification = None
    if parsed.opening_balance and parsed.closing_balance:
        opening = _parse_amount(parsed.opening_balance)
        closing = _parse_amount(parsed.closing_balance)
        credits = _parse_amount(parsed.credit_total or "0")
        debits = _parse_amount(parsed.debit_total or "0")
        computed_closing = opening + credits - debits
        delta = closing - computed_closing
        balance_verification = {
            "opening_balance": parsed.opening_balance,
            "closing_balance": parsed.closing_balance,
            "computed_closing": f"{computed_closing:,.2f}",
            "delta": f"{delta:,.2f}",
            "is_balanced": abs(delta) < Decimal("1"),
        }

    return {
        "matched": matched,
        "missing": missing,
        "balance_verification": balance_verification,
        "debit_total": parsed.debit_total,
        "credit_total": parsed.credit_total,
        "opening_balance": parsed.opening_balance,
        "closing_balance": parsed.closing_balance,
    }


_GENERIC_COUNTERPARTIES = {"payment received", "payment successful", "payment done"}


async def enrich_matched_transactions(recon: dict) -> int:
    """Update DB transaction counterparty from statement for matched transactions.

    Prefers the parser-derived `counterparty` (clean merchant/beneficiary
    pulled out of structured narrations like UPI/MMT/IMPS) and falls back
    to the raw narration when the parser couldn't extract one.
    """
    enriched = 0
    async with async_session() as session:
        for entry in recon.get("matched", []):
            counterparty = (entry.get("counterparty") or "").strip()
            narration = (entry.get("narration") or "").strip()
            new_value = counterparty or narration
            if not new_value:
                continue

            db_txn_id = entry.get("db_txn_id")
            if not db_txn_id:
                continue

            txn = await session.get(Transaction, db_txn_id)
            if not txn:
                continue

            existing = (txn.counterparty or "").strip()
            if existing and existing.lower() not in _GENERIC_COUNTERPARTIES:
                continue

            txn.counterparty = new_value
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


# ---------------------------------------------------------------------------
# Sync PDF parsing helper
# ---------------------------------------------------------------------------


def _parse_pdf_bytes_sync(pdf_bytes: bytes, bank: str, password: str | None = None):
    """Save PDF bytes to temp file, parse, and clean up."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = Path(f.name)
    try:
        return parse_bank_statement(tmp_path, bank, password)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Account lookup
# ---------------------------------------------------------------------------


def _last4(account_number: str | None) -> str | None:
    """Extract last 4 digits from an account number string."""
    if not account_number:
        return None
    digits = re.sub(r"[^0-9]", "", account_number)
    return digits[-4:] if len(digits) >= 4 else (digits if digits else None)


async def _find_bank_account(bank: str, parsed) -> "Account | None":
    """Find an existing bank_account Account. Returns None when no match;
    statements do not auto-create accounts.
    """
    stmt_acct_number = parsed.account_number
    stmt_last4 = _last4(stmt_acct_number)

    async with async_session() as session:
        bank_accounts = (
            (
                await session.execute(
                    select(Account).where(
                        Account.bank == bank,
                        Account.type == "bank_account",
                        Account.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )

    # Try to match by account number
    account = None
    if stmt_last4:
        for acc in bank_accounts:
            if _last4(acc.account_number) == stmt_last4:
                account = acc
                break
    elif stmt_acct_number:
        # Full or partial match
        for acc in bank_accounts:
            if acc.account_number and stmt_acct_number in acc.account_number:
                account = acc
                break

    # Fallback: if exactly one bank_account for this bank, use it
    if not account and len(bank_accounts) == 1:
        account = bank_accounts[0]

    if account:
        return account

    logger.info(
        "No matching bank_account for bank=%s account=%s; statement not imported",
        bank,
        stmt_acct_number,
    )
    return None


# ---------------------------------------------------------------------------
# End-to-end email processing
# ---------------------------------------------------------------------------


async def process_bank_statement_email(
    bank: str,
    raw_bytes: bytes,
    email_subject: str,
    source_id: int | None = None,
    password_hint: str | None = None,
) -> dict | None:
    """Try to process an email as a bank account statement.

    Returns a dict with bank_statement_upload_id and stats if successful, None otherwise.
    """
    subject_lower = (email_subject or "").lower()

    # Must contain "statement"
    if "statement" not in subject_lower:
        return None

    # Accept bank account statements: "account statement" without "card"
    # Also accept generic "statement" that doesn't look like a CC statement
    is_bank_stmt = "account statement" in subject_lower and "card" not in subject_lower
    is_cc_stmt = any(
        kw in subject_lower for kw in ("credit card", "card statement", "cc statement")
    )
    if is_cc_stmt:
        return None  # Let the CC handler deal with it
    if not is_bank_stmt and "statement" in subject_lower:
        # Ambiguous — we'll try parsing and see if the PDF looks like a bank statement
        pass

    # Require at least one bank account for this bank; statements must not
    # auto-create accounts.
    async with async_session() as session:
        has_bank_account = (
            await session.execute(
                select(Account.id).where(
                    Account.bank == bank,
                    Account.type == "bank_account",
                    Account.active.is_(True),
                )
            )
        ).first() is not None
    if not has_bank_account:
        logger.info("Skipping bank statement path: no bank_account for bank=%s", bank)
        return None

    # Extract PDF attachments
    pdfs = extract_pdf_from_email(raw_bytes)
    if not pdfs:
        logger.info(
            "Bank statement email has no PDF attachment: bank=%s subject=%r",
            bank,
            email_subject[:80] if email_subject else "",
        )
        return None

    filename, pdf_bytes = pdfs[0]
    logger.info(
        "Found PDF in bank statement email: bank=%s file=%s (%d bytes)",
        bank,
        filename,
        len(pdf_bytes),
    )

    # Use pre-extracted hint from parse_email(), fall back to direct extraction
    if not password_hint:
        password_hint = extract_password_hint(raw_bytes, bank=bank)
    if password_hint:
        logger.info("Password hint: %s", password_hint)

    # Parse the PDF
    parsed = None
    try:
        parsed = await asyncio.to_thread(_parse_pdf_bytes_sync, pdf_bytes, bank)
    except ValueError as e:
        if "encrypt" not in str(e).lower() and "password" not in str(e).lower():
            logger.warning("Failed to parse bank statement PDF: %s", e)
            raise BankStatementProcessingError(
                f"Failed to parse {bank} statement PDF {filename!r}: {e}"
            ) from e

        # PDF is encrypted — try stored passwords
        fernet = get_fernet()
        async with async_session() as session:
            bank_accounts = (
                (
                    await session.execute(
                        select(Account).where(
                            Account.bank == bank,
                            Account.type == "bank_account",
                            Account.active.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )

        passwords_to_try = []
        accounts_without_password = []
        for acc in bank_accounts:
            if not acc.statement_password:
                accounts_without_password.append(acc.label)
                continue
            try:
                pw = fernet.decrypt(acc.statement_password.encode()).decode()
                passwords_to_try.append((acc, pw))
            except Exception as e:
                logger.warning(
                    "Failed to decrypt stored statement_password for %s (%s): %s",
                    bank,
                    acc.label,
                    e,
                )
        logger.info(
            "Encrypted PDF: %d/%d bank accounts have a stored password for bank=%s (no password: %s)",
            len(passwords_to_try),
            len(bank_accounts),
            bank,
            accounts_without_password or "none",
        )

        for acc, pw in passwords_to_try:
            try:
                parsed = await asyncio.to_thread(
                    _parse_pdf_bytes_sync, pdf_bytes, bank, pw
                )
                logger.info(
                    "Decrypted bank statement PDF using stored password for %s (%s)",
                    bank,
                    acc.label,
                )
                break
            except Exception as e:
                logger.info(
                    "Stored password for %s (%s) did not unlock PDF: %s",
                    bank,
                    acc.label,
                    e,
                )
                continue

        if not parsed:
            # Save for manual retry
            STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            safe_name = _safe_filename(filename)
            file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
            file_path.write_bytes(pdf_bytes)

            # Only attach if there's exactly one bank account for this bank;
            # otherwise we'd mis-attribute the PDF and (worse) leak the hint to
            # the wrong account.
            account = bank_accounts[0] if len(bank_accounts) == 1 else None
            if not account:
                logger.warning(
                    "Encrypted bank statement received but %d candidate accounts for bank=%s — leaving unassigned (%s)",
                    len(bank_accounts),
                    bank,
                    safe_name,
                )
            if account:
                async with async_session() as session:
                    upload = BankStatementUpload(
                        account_id=account.id,
                        bank=bank,
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
                        "Encrypted bank statement saved for manual password entry: %s",
                        safe_name,
                    )
                    return {
                        "bank_statement_upload_id": upload.id,
                        "matched": 0,
                        "missing": 0,
                        "imported": 0,
                    }
            raise BankStatementProcessingError(
                f"Encrypted {bank} statement PDF could not be decrypted with any "
                f"stored password and {len(bank_accounts)} candidate accounts exist "
                f"for this bank — leaving unassigned ({safe_name})"
            )
    except BankStatementProcessingError:
        raise
    except Exception as e:
        logger.exception("Failed to parse bank statement PDF")
        raise BankStatementProcessingError(
            f"Unexpected error processing {bank} statement PDF {filename!r}: {e}"
        ) from e

    # Verify it looks like a bank account statement (not a CC statement)
    # If the parsed result has no transactions, bail
    if not parsed.transactions:
        logger.info("Bank statement parsing returned no transactions for %s", filename)
        return None

    account = await _find_bank_account(bank, parsed)
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

    recon = reconcile_bank_statement(parsed, db_txns, account.id)

    # Save the PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_filename(filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    file_path.write_bytes(pdf_bytes)

    # Create BankStatementUpload and import missing transactions
    async with async_session() as session:
        # Store password hint on the account if we found one
        if password_hint:
            acct_row = await session.get(Account, account.id)
            if acct_row and not acct_row.statement_password_hint:
                acct_row.statement_password_hint = password_hint

        upload = BankStatementUpload(
            account_id=account.id,
            bank=parsed.bank or bank,
            filename=safe_name,
            file_path=str(file_path),
            status="parsed",
            account_number=parsed.account_number,
            account_holder_name=parsed.account_holder_name,
            opening_balance=parsed.opening_balance,
            closing_balance=parsed.closing_balance,
            statement_period_start=parsed.statement_period_start,
            statement_period_end=parsed.statement_period_end,
            parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
            matched_count=len(recon["matched"]),
            missing_count=len(recon["missing"]),
            reconciliation_data=reconciliation_to_json(recon),
        )
        session.add(upload)
        await session.flush()

        # Auto-import all missing transactions. Each row runs inside its own
        # SAVEPOINT so a single duplicate / constraint violation cannot abort
        # the entire statement import. The upload row stays committed with
        # whatever subset succeeded; failures are tagged on the missing
        # entries so the UI can show them.
        link_ctx = await build_link_context(session)

        imported = 0
        duplicate_count = 0
        import_error_count = 0
        imported_txns: list[tuple[int, dict]] = []
        for entry in recon["missing"]:
            try:
                amount = _parse_amount(entry["amount"])
                txn_date = _parse_date(entry["date"])
            except ValueError, KeyError, InvalidOperation:
                entry["import_error"] = "could not parse amount/date"
                continue

            txn = Transaction(
                bank_statement_upload_id=upload.id,
                account_id=account.id,
                bank=bank,
                email_type="bank_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("counterparty") or entry.get("narration"),
                account_mask=_last4(parsed.account_number),
                reference_number=entry.get("reference_number"),
                channel=entry.get("channel") or "bank_statement",
                raw_description=entry.get("narration"),
            )

            try:
                async with session.begin_nested():
                    session.add(txn)
                    await session.flush()
                    link_transaction(link_ctx, txn)
                    await session.flush()
            except IntegrityError as e:
                duplicate_count += 1
                entry["duplicate"] = True
                entry["import_error"] = "duplicate transaction"
                logger.info(
                    "Skipping duplicate %s stmt txn (ref=%s direction=%s "
                    "date=%s amount=%s): %s",
                    bank,
                    entry.get("reference_number"),
                    entry["direction"],
                    entry["date"],
                    entry["amount"],
                    e.orig,
                )
                continue
            except Exception as e:
                import_error_count += 1
                entry["import_error"] = f"{type(e).__name__}: {e}"
                logger.exception(
                    "Unexpected error importing stmt txn (ref=%s)",
                    entry.get("reference_number"),
                )
                continue

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
            upload.status = "imported"
        elif imported > 0:
            upload.status = "partial_import"
        if import_error_count or duplicate_count:
            details = []
            if duplicate_count:
                details.append(f"{duplicate_count} duplicate")
            if import_error_count:
                details.append(f"{import_error_count} unexpected error")
            upload.error = (
                f"Skipped {', '.join(details)} row(s) during auto-import; "
                "see reconciliation details."
            )
        await session.commit()
        upload_id = upload.id

    # Notifications and enrichment run outside the DB session so that network
    # I/O doesn't hold the session open.
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
                source="bank_statement",
                txns=imported_txns,
            )

    enriched = await enrich_matched_transactions(recon)

    logger.info(
        "Processed bank statement email: bank=%s account=%s matched=%d missing=%d "
        "imported=%d duplicates=%d errors=%d enriched=%d",
        bank,
        account.label,
        len(recon["matched"]),
        len(recon["missing"]),
        imported,
        duplicate_count,
        import_error_count,
        enriched,
    )
    return {
        "bank_statement_upload_id": upload_id,
        "matched": len(recon["matched"]),
        "missing": len(recon["missing"]),
        "imported": imported,
        "duplicates": duplicate_count,
        "import_errors": import_error_count,
        "enriched": enriched,
    }
