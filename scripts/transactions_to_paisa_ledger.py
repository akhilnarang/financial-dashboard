#!/usr/bin/env python3
"""Render a Paisa-format Ledger journal from any financial-dashboard SQLite DB.

Reads the `transactions` (and `accounts`) table directly and emits journal
entries via the real production renderer (`financial_dashboard.services.paisa.
render_paisa_journal_entry`), so the output matches exactly what paisa-mode
ingestion would produce.

Scope by default: only alert-derived rows (email_type NOT IN bank_statement,
cc_statement) — i.e. what paisa-mode would actually export. Use
`--include-statements` to render statement-derived rows too (warning: those
overlap with alerts for shared months and will double-count).

Usage (from the financial-dashboard repo root):

    uv run python scripts/transactions_to_paisa_ledger.py
    uv run python scripts/transactions_to_paisa_ledger.py --db <path> --out <path>
    uv run python scripts/transactions_to_paisa_ledger.py --include-statements

Defaults: --db paisa_test.db, --out paisa_test.ledger (both at repo root, gitignored).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
from decimal import Decimal
from pathlib import Path

from financial_dashboard.db import PaisaExport
from financial_dashboard.services.paisa import render_paisa_journal_entry

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "paisa_test.db"
DEFAULT_OUT = REPO_ROOT / "paisa_test.ledger"

STATEMENT_KINDS = ("bank_statement", "cc_statement")


def _resolve_source_account(
    acct_type: str | None, acct_label: str | None, card_mask: str | None
) -> tuple[str, bool]:
    """(account, missing_map) — prefer the linked Account so the journal shows
    real names. Fall back by card-mask presence the way the spec's resolver
    does in real paisa mode (no account_map configured)."""
    if acct_type == "credit_card" and acct_label:
        return f"Liabilities:CreditCard:{acct_label}", False
    if acct_type == "bank_account" and acct_label:
        return f"Assets:Bank:{acct_label}", False
    if card_mask:
        return "Liabilities:Unknown", True
    return "Assets:Unknown", True


def _parse_time(value: str | None) -> dt.time | None:
    if not value:
        return None
    try:
        return dt.time.fromisoformat(str(value).split(".")[0])
    except ValueError:
        return None


def render(db_path: Path, out_path: Path, include_statements: bool) -> tuple[int, int]:
    where = "" if include_statements else "WHERE t.email_type NOT IN (?, ?)"
    params: tuple = () if include_statements else STATEMENT_KINDS

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"""
        SELECT t.id, t.bank, t.transaction_date, t.transaction_time, t.direction,
               t.amount, t.currency, t.counterparty, t.card_mask,
               t.account_mask, t.reference_number, t.email_type, t.source,
               t.sms_message_id, t.email_id,
               a.type AS acct_type, a.label AS acct_label
        FROM transactions t
        LEFT JOIN accounts a ON a.id = t.account_id
        {where}
        ORDER BY t.transaction_date, t.transaction_time, t.id
        """,
        params,
    ).fetchall()

    exports: list[PaisaExport] = []
    missing = 0
    for r in rows:
        source_account, miss = _resolve_source_account(
            r["acct_type"], r["acct_label"], r["card_mask"]
        )
        missing += miss
        counterparty_account = (
            "Expenses:Uncategorized"
            if r["direction"] == "debit"
            else "Income:Uncategorized"
        )
        exports.append(
            PaisaExport(
                id=r["id"],
                source=r["source"] or "email",
                email_id=r["email_id"],
                sms_message_id=r["sms_message_id"],
                idempotency_key=f"db:{r['id']}",
                bank=r["bank"],
                email_type=r["email_type"],
                direction=r["direction"],
                amount=Decimal(str(r["amount"])).quantize(Decimal("0.01")),
                currency=(r["currency"] or "INR").upper(),
                transaction_date=(
                    dt.date.fromisoformat(r["transaction_date"])
                    if r["transaction_date"]
                    else None
                ),
                transaction_time=_parse_time(r["transaction_time"]),
                counterparty=r["counterparty"] or r["email_type"],
                reference_number=r["reference_number"],
                card_mask=r["card_mask"],
                account_mask=r["account_mask"],
                source_account=source_account,
                counterparty_account=counterparty_account,
                missing_account_mapping=bool(miss),
                status="exported",
            )
        )

    body = "\n".join(render_paisa_journal_entry(e).rstrip("\n") for e in exports)
    if body and not body.endswith("\n"):
        body += "\n"

    scope = "all rows" if include_statements else "alert-derived only"
    header = (
        f"; Generated via financial_dashboard.services.paisa renderer ({scope}).\n"
        f"; {len(exports)} entries. {missing} hit Liabilities/Assets:Unknown fallback.\n"
        "; Include from your Paisa main journal.\n\n"
    )

    fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(header + body)
    os.chmod(out_path, 0o600)
    return len(exports), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"source SQLite DB (default: {DEFAULT_DB})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output journal file (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--include-statements",
        action="store_true",
        help="also render bank_statement / cc_statement rows (will double-count vs alerts for shared months)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"DB not found: {args.db}")

    count, missing = render(args.db, args.out, args.include_statements)
    print(f"wrote {count} entries to {args.out} ({missing} fallback to Unknown)")


if __name__ == "__main__":
    main()
