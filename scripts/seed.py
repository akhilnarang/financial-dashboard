#!/usr/bin/env python
"""Seed the database with known fetch rules for all supported banks.

Idempotent -- safe to run multiple times. Skips rules that already exist.
Does NOT set up email source credentials -- add those via the web UI at /sources.

On first run, generates a Fernet encryption key and writes it to .env if one
doesn't already exist.

Usage:
    uv run python seed.py
"""

import asyncio
from pathlib import Path

from sqlalchemy import select

from financial_dashboard.db import EmailSource, FetchRule, init_db, async_session


def _ensure_fernet_key():
    """Generate EMAIL_SOURCE_MASTER_KEY and write to .env if not already set."""
    from financial_dashboard.config import Settings

    if Settings().email_source_master_key:
        return

    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()

    env_path = Path(".env")
    with env_path.open("a") as f:
        f.write(f"EMAIL_SOURCE_MASTER_KEY={key}\n")
    print("Generated Fernet key and wrote to .env")


# (provider, bank, sender, subject_filter, folder, email_kind)
# email_kind: one of "transaction" (txn alert), "cc_statement" (CC PDF),
# "bank_statement" (bank PDF), or "statement" (legacy, try both pipelines).
# None means auto-detect (txn parse, fall back to both statement pipelines).
RULES = [
    # Slice (Gmail)
    ("gmail", "slice", "noreply@slice.bank.in", None, None, None),
    ("gmail", "slice", "noreply@sliceit.com", None, None, None),
    # Slice CC + bank e-statements share the same sender; disambiguate by subject.
    (
        "gmail",
        "slice",
        "noreply@slice.bank.in",
        "credit card statement",
        None,
        "cc_statement",
    ),
    (
        "gmail",
        "slice",
        "noreply@slice.bank.in",
        "bank statement",
        None,
        "bank_statement",
    ),
    # ICICI (Gmail)
    ("gmail", "icici", "credit_cards@icicibank.com", None, None, None),
    ("gmail", "icici", "customernotification@icici.bank.in", None, None, None),
    ("gmail", "icici", "customercare@icicibank.com", "Transaction alert", None, None),
    ("gmail", "icici", "credit_cards@icici.bank.in", None, None, None),
    (
        "gmail",
        "icici",
        "credit_cards@icici.bank.in",
        "credit card statement",
        None,
        "cc_statement",
    ),
    ("gmail", "icici", "estatement@icici.bank.in", None, None, "bank_statement"),
    # HDFC (Gmail)
    ("gmail", "hdfc", "alerts@hdfcbank.bank.in", None, None, None),
    (
        "gmail",
        "hdfc",
        "Emailstatements.cards@hdfcbank.net",
        "statement",
        None,
        "cc_statement",
    ),
    (
        "gmail",
        "hdfc",
        "hdfcbanksmartstatement@hdfcbank.bank.in",
        None,
        None,
        "bank_statement",
    ),
    # Axis (Gmail)
    ("gmail", "axis", "alerts@axis.bank.in", None, None, None),
    ("gmail", "axis", "cc.statements@axis.bank.in", "statement", None, "cc_statement"),
    # IndusInd (Gmail)
    ("gmail", "indusind", "transactionalert@indusind.com", None, None, None),
    ("gmail", "indusind", "indusind_bank@indusind.com", "Transaction", None, None),
    ("gmail", "indusind", "IndusInd_Bank@indusind.com", "Transaction", None, None),
    (
        "gmail",
        "indusind",
        "indusind_bank@indusind.com",
        "Payment Confirmation",
        None,
        None,
    ),
    (
        "gmail",
        "indusind",
        "IndusInd_Bank@indusind.com",
        "Payment Confirmation",
        None,
        None,
    ),
    (
        "gmail",
        "indusind",
        "creditcard.estatements@indusind.com",
        "statement",
        None,
        "cc_statement",
    ),
    # Kotak (Gmail)
    ("gmail", "kotak", "BankAlerts@kotak.com", None, None, None),
    ("gmail", "kotak", "no-reply@kotak.com", None, None, None),
    ("gmail", "kotak", "nach.alerts@kotak.bank.in", None, None, None),
    (
        "gmail",
        "kotak",
        "BankStatements@kotak.bank.in",
        "statement",
        None,
        "bank_statement",
    ),
    # SBI Card (Gmail)
    ("gmail", "sbi", "onlinesbicard@sbicard.com", None, None, None),
    ("gmail", "sbi", "paynet@billdesk.in", None, None, None),
    ("gmail", "sbi", "Statements@sbicard.com", "statement", None, "cc_statement"),
    # HSBC (Gmail)
    ("gmail", "hsbc", "hsbc@mail.hsbc.co.in", "Credit Card", None, None),
    ("gmail", "hsbc", "alerts@mail.hsbc.co.in", "Credit Card", None, None),
    (
        "gmail",
        "hsbc",
        "creditcardstatement@mail.hsbc.co.in",
        "statement",
        None,
        "cc_statement",
    ),
    (
        "gmail",
        "hsbc",
        "campaign@mail.hsbc.co.in",
        "statement",
        None,
        "cc_statement",
    ),
    # IDFC FIRST (Gmail) — one sender handles both CC and bank statements;
    # disambiguate by subject.
    ("gmail", "idfc", "transaction.alerts@idfcfirstbank.com", None, None, None),
    ("gmail", "idfc", "noreply@idfcfirstbank.com", None, None, None),
    (
        "gmail",
        "idfc",
        "statement@idfcfirst.bank.in",
        "credit card statement",
        None,
        "cc_statement",
    ),
    (
        "gmail",
        "idfc",
        "statement@idfcfirst.bank.in",
        "account statement",
        None,
        "bank_statement",
    ),
    # Equitas (Gmail)
    ("gmail", "equitas", "cc-alerts@equitas.bank.in", None, None, None),
    # OneCard / BOBCARD (Gmail)
    ("gmail", "onecard", "no-reply@getonecard.app", None, None, None),
    ("gmail", "onecard", "statement@getonecard.app", "statement", None, "cc_statement"),
    # Union Bank of India (Gmail)
    (
        "gmail",
        "uboi",
        "noreplyunionbankofindia@unionbankofindia.bank.in",
        None,
        None,
        None,
    ),
    # Bank of Maharashtra (Gmail)
    ("gmail", "bom", "mahaalert@mahabank.co.in", None, None, None),
    # Yes Bank (Gmail)
    ("gmail", "yesbank", "alerts@yes.bank.in", "Transaction Alert", None, None),
    # Jupiter — UPI txn alerts from alerts@, CC statements from noreply@.
    ("gmail", "jupiter", "alerts@jupiter.money", None, None, None),
    (
        "gmail",
        "jupiter",
        "noreply@jupiter.money",
        "credit card statement",
        None,
        "cc_statement",
    ),
]


async def main():
    await init_db()
    async with async_session() as session:
        existing = await session.execute(select(FetchRule))
        existing_rules = existing.scalars().all()
        existing_keys = {(r.provider, r.sender, r.subject) for r in existing_rules}
        # Build lookup for updating email_kind on existing rules
        existing_by_key = {(r.provider, r.sender, r.subject): r for r in existing_rules}

        # Build seed email_kind lookup
        seed_kinds = {
            (provider, sender, subject): email_kind
            for provider, bank, sender, subject, folder, email_kind in RULES
        }

        added = 0
        for provider, bank, sender, subject, folder, email_kind in RULES:
            if (provider, sender, subject) in existing_keys:
                continue
            session.add(
                FetchRule(
                    provider=provider,
                    bank=bank,
                    sender=sender,
                    subject=subject,
                    folder=folder,
                    email_kind=email_kind,
                    enabled=True,
                )
            )
            added += 1

        # Backfill email_kind on existing rules that have NULL but seed specifies a value
        updated_kind = 0
        for key, email_kind in seed_kinds.items():
            if email_kind and key in existing_by_key:
                rule = existing_by_key[key]
                if rule.email_kind is None:
                    rule.email_kind = email_kind
                    updated_kind += 1

        await session.commit()
        total = len(existing_keys) + added
        print(
            f"Added {added} rules ({len(existing_keys)} already existed, {total} total)"
        )
        if updated_kind:
            print(f"Updated email_kind on {updated_kind} existing rules")

    # Auto-assign source_id to rules that don't have one
    async with async_session() as session:
        sources = {
            s.provider: s.id
            for s in (await session.execute(select(EmailSource))).scalars().all()
        }
        if sources:
            orphan_rules = (
                (
                    await session.execute(
                        select(FetchRule).where(FetchRule.source_id.is_(None))
                    )
                )
                .scalars()
                .all()
            )
            linked = 0
            for rule in orphan_rules:
                if rule.provider in sources:
                    rule.source_id = sources[rule.provider]
                    linked += 1
            await session.commit()
            if linked:
                print(f"Auto-linked {linked} rules to existing sources")
        else:
            print()
            print(
                "NOTE: No email sources found. Add them at /sources, then re-run seed.py to auto-link."
            )


if __name__ == "__main__":
    _ensure_fernet_key()
    asyncio.run(main())
