"""Fixed constants for the synthetic seed generator.

Everything here is a plain literal so the pure generator modules
(:mod:`scripts.synth.scenario`, :mod:`scripts.synth.paisa`) never import the
dashboard runtime — that keeps generation deterministic, side-effect free, and
unit-testable without a database or a running app.

The string literals below mirror the dashboard's ``db.enums`` StrEnum values
and the categorization vocabulary's seed slugs. The loader
(:mod:`scripts.synth.loader`) is the only module that imports production code;
it validates these literals against the real enums so drift surfaces as a
test failure rather than a silent mismatch.
"""

import datetime as _dt
import uuid

#: Bumped when the generated artefact shape changes in a way that invalidates
#: an existing manifest. ``verify`` refuses a manifest whose generator version
#: does not match the running code.
#:
#: 1.2.0 — realism/fidelity pass: realistic monthly inflows (interest, other
#: income, repayment/transfers-in, refunds/cashback), corrected repayment
#: polarity, profile-scaled expense bands + opening balances so bank-scope
#: income/expense >= 1.1 (smoke/ci) / >= 1.0 (stress), nonnegative active-bank
#: running balances, dates clamped <= as_of, CC due-date in DD/MM/YYYY,
#: balance-derived snapshots, statement-linked rows, pending/failed/skipped
#: emails + review statuses, fidelity category stamping, bulk sms_message_id +
#: SmsMessage.transaction_id reverse link.
#:
#: 1.3.0 — canonical scenario/manifest expansion: explicit scenario-branch IDs
#: + coverage metadata (grouped merge/link, categorization, statement recon,
#: refunds/reversals, FX, networth/CAS/manual, workflow/projection) recorded in
#: the manifest and policed by verify; multiple CC + bank statement uploads
#: across paid/partial/unpaid/parse-error statuses; distinct merchant refund /
#: cashback / fee-reversal / CC-refund-credit / card-side-payment / transfer-in
#: repayment / investment-redemption edges; USD/EUR/GBP FX pairs with separated
#: merchant/category/reference metadata plus invalid + blank currency edges;
#: blank counterparty/category, AM/PM pair, ref-mismatch pair, balance-conflict
#: split, orphan CC payment; category_method variation
#: (manual/rule/llm/pending_llm/synthetic) + review_status with reason /
#: confidence / model; multiple-PAN CAS with reconciled + unreconciled
#: portfolios and complete/disposal/incomplete lot shapes; multi-month
#: snapshots for the net-worth trend; count_rows no longer swallows table
#: errors.
#:
#: 1.3.1 — projection-realism pass: truthful opening-balance snapshots at the
#: documented ``PROJECTION_CUTOVER`` (2025-07-01) for every selected
#: asset/liability account, replayed from the scenario's tracked balances so the
#: projected running balance never goes negative; economically-ordered
#: investment contributions (monthly) ahead of a smaller redemption so
#: ``Assets:Investments:Unallocated`` never goes negative; at least one
#: bank-side card payment carries an explicit ``card_id`` resolving to a
#: selected card liability (plus a deliberately unresolved one); the alleged
#: invalid ``XXX`` edge replaced with a genuinely projection-invalid token
#: (``000``, digit-first → ``invalid_currency``); the complete CAS lot's
#: instrument no longer carries a redemption (so it is no longer suppressed and
#: ``investment_lot_count`` >= 1).
GENERATOR_VERSION = "1.3.1"

#: Bumped when the on-disk manifest JSON schema changes.
SCHEMA_VERSION = "1"

#: Stable UUID namespace for UUIDv5 stable ids. Hard-coded so two runs on two
#: machines produce identical ids for the same ``(seed, profile, as_of)``.
SYNTH_NAMESPACE = uuid.UUID("c4d2e1f0-5b6a-4c7d-8e9f-0a1b2c3d4e5f")

#: Default root for all synthetic output. Always under ``data/`` (which is
#: gitignored) so generated volume is never committed. The loader refuses to
#: touch any path that escapes this root.
DEFAULT_SYNTHETIC_ROOT = "data/synthetic"

#: The default profile used by ``generate`` and ``load`` when none is passed.
DEFAULT_PROFILE = "smoke"

#: The default seed. Stable so a bare ``uv run python -m scripts.synth generate``
#: is reproducible.
DEFAULT_SEED = 4242

#: The canonical projection cutover the synthetic corpus is shaped for. The
#: scenario seeds truthful opening-balance snapshots at this date (one per
#: selected asset/liability account, value = the account's replayed balance at
#: the cutover) so a projection run with ``cutover_date = PROJECTION_CUTOVER``
#: never drives any account's running balance negative.
PROJECTION_CUTOVER = _dt.date(2025, 7, 1)

#: A currency token the projection genuinely invalidates (its sanitizer rejects
#: a value whose cleaned form does not start with a letter — see
#: ``projection._normalize_fx_currency``). Used in place of ``XXX`` (which is a
#: valid ISO 4217 placeholder and so reports ``missing_fx_rate`` rather than
#: ``invalid_currency``). ``"000"`` cleans to ``"000"`` (digit-first) → invalid.
INVALID_CURRENCY_TOKEN = "000"

#: The literal confirmation flag the destructive ``reset`` command requires.
#: Long and explicit on purpose: a typo or a shell-glob must not be able to
#: trip a wipe.
RESET_CONFIRMATION_FLAG = "yes-delete-the-synthetic-db"

#: Bulk-lane transactions take primary keys from this base so they never
#: collide with the autoincrement ids the fidelity lane's
#: ``merge_transaction`` assigns (which start at 1).
BULK_TXN_ID_BASE = 1_000_000


# ---------------------------------------------------------------------------
# Enum-equivalent string literals (mirror financial_dashboard.db.enums)
# ---------------------------------------------------------------------------

BANK_ACCOUNT = "bank_account"
CREDIT_CARD = "credit_card"

DIRECTION_DEBIT = "debit"
DIRECTION_CREDIT = "credit"

SNAPSHOT_ASSET = "asset"
SNAPSHOT_LIABILITY = "liability"

SNAP_CAT_BANK_BALANCE = "bank_balance"
SNAP_CAT_CC_OUTSTANDING = "cc_outstanding"
SNAP_CAT_INVESTMENT = "investment"
SNAP_CAT_MANUAL_ASSET = "manual_asset"
SNAP_CAT_MANUAL_LIABILITY = "manual_liability"

SNAP_SOURCE_BANK = "bank_statement"
SNAP_SOURCE_CC = "cc_statement"
SNAP_SOURCE_CAS = "cas"
SNAP_SOURCE_MANUAL = "manual"

MANUAL_ASSET = "asset"
MANUAL_LIABILITY = "liability"

MANUAL_CAT_PROPERTY = "property"
MANUAL_CAT_EPF_PPF = "epf_ppf"
MANUAL_CAT_GOLD = "gold"
MANUAL_CAT_CASH = "cash"
MANUAL_CAT_LOAN = "loan"
MANUAL_CAT_OTHER = "other"

DEPOSITORY_NSDL = "nsdl"
DEPOSITORY_CDSL = "cdsl"

PAYMENT_UNPAID = "unpaid"
PAYMENT_PAID = "paid"
PAYMENT_PARTIAL = "partial"
PAYMENT_LATE = "late"
PAYMENT_ZERO = "zero"

#: Statement upload lifecycle statuses (mirror the production
#: ``StatementUpload.status`` literals).
STMT_STATUS_PARSED = "parsed"
STMT_STATUS_PARTIAL = "partial"
STMT_STATUS_PASSWORD_REQUIRED = "password_required"
STMT_STATUS_PARSE_ERROR = "parse_error"
STMT_STATUS_IMPORTED = "imported"

#: Statement ``source_kind`` literals.
STMT_SOURCE_PDF = "pdf"
STMT_SOURCE_EMAIL_SUMMARY = "email_summary"

EMAIL_KIND_TXN = "transaction"
EMAIL_KIND_CC_STATEMENT = "cc_statement"
EMAIL_KIND_BANK_STATEMENT = "bank_statement"
EMAIL_KIND_CAS = "cas_statement"

#: Subset of the categorization vocabulary's seed slugs that the generator
#: assigns deterministically. Kept in sync with
#: ``services.categorization.vocabulary.SEED_CATEGORIES``; the loader asserts
#: membership so a stray slug fails loudly.
SEED_CATEGORY_SLUGS: tuple[str, ...] = (
    "salary",
    "interest",
    "refund",
    "cashback_rewards",
    "other_income",
    "repayment",
    "expense",
    "investment",
    "investment_redemption",
    "self_transfer",
    "credit_card_payment",
    "bill_payment",
    "groceries",
    "dining",
    "fuel",
    "car_maintenance",
    "transport",
    "shopping",
    "utilities",
    "subscriptions",
    "rent",
    "emi_loan",
    "insurance",
    "healthcare",
    "entertainment",
    "travel",
    "education",
    "personal_care",
    "fees_charges",
    "tax",
    "cash_withdrawal",
    "charity_gift",
    "gift",
    "misc",
    "unknown",
)
