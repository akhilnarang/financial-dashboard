# financial-dashboard

Self-hosted personal finance service that fetches bank transaction alert emails from Gmail and Fastmail, parses them into structured transactions, reconciles credit card statements, and provides a web dashboard for viewing and managing your financial data.

## Tech Stack

- **FastAPI** + Jinja2 templates + [oat.ink](https://oat.ink) CSS
- **SQLAlchemy** async + **SQLite** (aiosqlite)
- **bank-email-parser** library for email parsing (12 Indian banks, 28+ email formats)
- **cc-parser** library for CC statement PDF parsing and reconciliation
- **Fernet** symmetric encryption for stored email credentials and statement passwords
- Gmail via IMAP, Fastmail via JMAP

## Quickstart

```bash
git clone https://github.com/AkhilNarang/financial-dashboard.git
cd financial-dashboard
mkdir -p data
uv sync --no-dev
uv run python scripts/seed.py   # generates .env with Fernet key + seeds fetch rules
uv run fastapi dev      # http://localhost:8000 (with auto-reload)
```

> **Warning:** There is currently no authentication on the web UI. Only run this on
> a trusted network or behind a reverse proxy with auth.

Once running:
1. Add email sources at `/sources` (Gmail app password or Fastmail API token)
2. Assign sources to rules at `/rules` (re-run `scripts/seed.py` after adding sources to auto-link)
3. Click "Poll Now" on the dashboard or wait for automatic polling every 15 minutes

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_SOURCE_MASTER_KEY` | (required) | Fernet key for encrypting credentials at rest. If unset, an ephemeral key is generated on each startup (credentials will not survive restarts). |
| `DB_URL` | `sqlite+aiosqlite:///./data/financial_dashboard.db` | SQLAlchemy database URL |
| `POLL_INTERVAL_MINUTES` | `15` | Automatic background polling interval |
| `POLL_FETCH_LIMIT_PER_RULE` | `50` | Max new emails fetched per rule per poll cycle |
| `TELEGRAM_BOT_TOKEN` | (optional) | Telegram bot token for real-time transaction notifications |
| `TELEGRAM_CHAT_ID` | (optional) | Telegram chat ID to send notifications to |

## Ledger Backend Modes

Use `ledger.backend` to choose how parsed SMS/email transaction alerts are consumed:

- `local` (default): existing dashboard-owned ledger. Alerts create/merge local `transactions` rows and continue through normal dashboard flows.
- `paisa`: ingestion bridge mode. Alerts are deduplicated into Paisa exports and written to a generated Ledger include journal. New alerts do **not** create local `transactions` rows.

### Paisa Mode (v1) Setup

Paisa mode uses these settings (`/settings`, category: `Ledger`):

- `ledger.backend` (default: `local`) -> set to `paisa` to enable bridge mode.
- `paisa.ledger_cli` (default: `ledger`) -> must be exactly `ledger` in v1. `hledger` and `beancount` are not supported by the current renderer and raise a config error.
- `paisa.main_journal_path` (default: empty) -> required absolute path to your main Paisa journal file.
- `paisa.generated_journal_path` (default: empty) -> required absolute path to the generated include file. It must be inside the main journal directory (or one of its subdirectories) and must not equal `paisa.main_journal_path`.
- `paisa.default_expense_account` (default: `Expenses:Uncategorized`)
- `paisa.default_income_account` (default: `Income:Uncategorized`)
- `paisa.fallback_asset_account` (default: `Assets:Unknown`)
- `paisa.fallback_liability_account` (default: `Liabilities:Unknown`)
- `paisa.account_map` (default: `{}`) -> JSON object mapping stable source IDs to Ledger account names.

Example include setup:

```ledger
# main journal: /home/you/finance/main.ledger
# generated include: /home/you/finance/imports/financial-dashboard.ledger
include imports/financial-dashboard.ledger
```

The generated file is rewritten from exported rows in deterministic order (`transaction_date`, `transaction_time`, `id`) whenever a new/export-enriched alert is processed.

### Paisa Account Mapping

`paisa.account_map` keys are stable source IDs derived from parsed bank + mask digits:

- `{bank}:card:{mask_digits}`
- `{bank}:account:{mask_digits}`

Example:

```json
{
  "hdfc:card:1234": "Liabilities:CreditCard:HDFC:1234",
  "hdfc:account:5678": "Assets:Bank:HDFC:5678"
}
```

If a source key is unmapped, exports still proceed:

- Counterparty side uses `paisa.default_expense_account` for debits and `paisa.default_income_account` for credits.
- Source side falls back to `paisa.fallback_liability_account` (card sources) or `paisa.fallback_asset_account` (account/no-mask sources).
- A `missing-map ...` comment is added to the journal entry so you can find gaps quickly (for example, `grep "missing-map"`).

### Paisa Mode Tradeoffs

Paisa mode is intentionally ingestion-only in v1:

- No new local transaction entries for parsed SMS/email alerts (so dashboard transaction pages do not receive new rows from those alerts).
- No local statement import/reconciliation pipeline for email statements, and statement emails are explicitly not imported in paisa mode.
- No local CC payment tracking/disambiguation hooks.
- No Telegram transaction/enrichment notifications for new ingested alerts in paisa mode.
- No Telegram reply workflow for transaction note/category updates from those alerts.

## Key Features

### Email Fetching
- **Gmail (IMAP)**: Connects via `imap.gmail.com` using an app password. Uses a two-phase fetch: Phase 0 searches by sender/subject/date criteria to collect UIDs, Phase 1 fetches lightweight headers and X-GM-MSGID for deduplication, Phase 2 fetches full RFC822 bodies only for new messages. Deduplicates across folders using X-GM-MSGID.
- **Fastmail (JMAP)**: Uses the Fastmail JMAP API. Queries email metadata first (including blobId), checks for existing remote IDs in the DB, then downloads only new message blobs.
- **Connection pooling**: All rules on the same email source are processed in a single provider connection (one IMAP session per Gmail source, one JMAP session per Fastmail source).
- **SINCE filtering**: On incremental polls, IMAP/JMAP queries include a date filter based on `last_synced_at` minus a 2-day margin to handle delayed delivery. New rules without a prior sync use a 3-month SINCE window for their initial backfill.
- **Backfill tracking**: Each `FetchRule` has an `initial_backfill_done_at` timestamp. Rules without this value perform a 3-month historical search on their first poll; the timestamp is set once the search phase completes successfully.

### Transaction Parsing
- Emails are parsed using **bank-email-parser**, which handles 12 Indian banks (Slice, ICICI, HDFC, Axis, IndusInd, Kotak, SBI, HSBC, IDFC FIRST, Equitas, OneCard, Union Bank of India) and 28+ email formats.
- Each parsed email produces a `Transaction` row with: bank, email type, direction (debit/credit), amount, currency, date, counterparty, card/account mask, reference number (UTR/UPI), channel, and available balance.
- Failed emails are saved to `financial_dashboard/data/failed/` as `.eml` files for debugging. Files older than 7 days are auto-cleaned.

### CC Statement Reconciliation
- **Automatic via email**: Statement emails (those with "statement" in the subject and a PDF attachment) are detected during polling. The PDF is extracted, parsed with **cc-parser**, and reconciled automatically.
- **Manual upload**: PDFs can be uploaded manually at `/statements` for any configured credit card account.
- **Reconciliation**: Statement transactions are matched to existing DB transactions by `(date, amount, direction)` with a ±1 day tolerance. Results are classified as matched, missing (in statement but not DB), or extra (in DB but not statement).
- **Auto-import**: Missing transactions are automatically imported as `Transaction` rows with `email_type="cc_statement"` and `channel="cc_statement"`.
- **Narration enrichment**: For matched transactions where the DB counterparty is null or a generic placeholder (e.g. "payment received"), the statement narration is written back to the `counterparty` field.
- **Password handling**: Encrypted PDFs are tried against all stored statement passwords for the bank. If none work, the PDF is saved to `financial_dashboard/data/statements/` with status `password_required` for manual retry via the UI. Passwords can be stored per-account (encrypted with Fernet) on the account edit page and will be used for future automated processing.
- **Auto-account creation**: If no matching credit card account is found for a statement's card number, a new Account (and Card) row is created automatically.

### Account and Card Management
- **Accounts** represent bank accounts, savings accounts, or credit cards. Each has a bank, label, type, and optional account number (last-4 or full number).
- **Cards** are physical cards linked to an account. An account can have multiple cards (primary + addon cards). Each card has a `card_mask` (e.g. `XX1234`).
- **Addon card support**: Multiple cards can be linked to a single credit card account (e.g. primary + spouse addon). The linker resolves transactions to the correct card and parent account.

### Transaction-to-Account Linking (`services/linker.py`)
Every transaction is auto-linked to an Account (and optionally a Card) using a four-level lookup cascade:

1. **card_mask -> cards table** — sets both `card_id` and `account_id`. Handles all mask formats (`XX1234`, `xx5678`, `XXXXXXX1234`, `0000 XXXX XXXX 1234`, `1234`, etc.) by extracting the last-4 digits.
2. **card_mask -> accounts table** — fallback for cards stored as Account rows (e.g. debit cards with `account_number` = last-4).
3. **account_mask -> accounts table** — for savings/current account masks.
4. **bank-only fallback** — links to the sole account for a bank, but only when exactly one account exists (avoids silent misattribution when a bank has both savings and CC accounts).

Linking is performed inline during polling and in batch via the `relink_orphans()` utility.

### Encrypted Credential Storage
- Email source credentials (Gmail app password, Fastmail API token) are encrypted with Fernet before storage.
- CC statement passwords are also stored Fernet-encrypted on the Account row.
- `EMAIL_SOURCE_MASTER_KEY` in `.env` is the Fernet key. Without it, a fresh ephemeral key is generated on each startup — stored credentials become unreadable across restarts.

### Telegram Notifications
- When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, the app sends real-time transaction notifications to a Telegram chat after each new transaction is parsed.
- Reply to a notification message to set a note on the transaction.

### Poll Status and Progress Reporting
- `GET /api/poll/status` returns a JSON object with `state` (idle/polling), `started_at`, `finished_at`, `last_stats`, `last_error`, and a `progress` dict (`{source, rule, email, detail}`) updated as each email is processed.
- The dashboard polls this endpoint to display live progress during a poll.

### Web UI
- **Dashboard** (`/`): Month-to-date stats (debit/credit/net flow, transaction count), operational stats (total emails, active rules), recent transactions, poll status and trigger.
- **Transactions** (`/transactions`): Paginated list (50/page) with filtering by bank, account, card, direction, and date range. Sortable by date, amount, bank, or counterparty. Clicking a row opens a detail modal.
- **Transaction Notes**: Each transaction has an editable note field that auto-saves via `POST /api/transactions/{id}/note`.
- **Original Email Viewer**: Re-fetches the raw email from the provider on demand and renders the HTML body in a sandboxed iframe with restrictive CSP headers.
- **Emails** (`/emails`): Last 200 fetched emails with status (pending/parsed/failed/skipped).
- **Accounts** (`/accounts`): CRUD for bank accounts and credit cards.
- **Email Sources** (`/sources`): CRUD for Gmail/Fastmail credentials. Test connectivity with `POST /api/sources/{id}/test`.
- **Rules** (`/rules`): CRUD for fetch rules (sender, subject, folder, bank, source assignment). Rules can be enabled/disabled individually.
- **Statements** (`/statements`): CC statement upload, reconciliation view, import controls, retry with password, and reprocess-failed-emails action.

## Architecture Overview

```
financial_dashboard/
  main.py         # FastAPI app factory and lifespan wiring
  api/            # JSON endpoints
  web/
    __init__.py   # Stable web router aggregation
    dashboard.py / accounts.py / sources.py / rules.py / transactions.py
    emails.py / statements.py / bank_statements.py / settings.py / polling.py
    forms.py
  services/       # Domain services
    statements/   # CC + bank statement subpackage
  schemas/        # Pydantic DTOs
  integrations/   # Parser + email provider adapters
  core/           # Shared templating, auth, crypto, date, and deps helpers
  db/             # Engine/session setup, models, enums, init_db glue
  templates/      # Jinja2 HTML templates
  static/         # CSS, JS
  data/
    failed/       # Failed email spool (.eml files, auto-cleaned after 7 days)
    statements/   # Saved CC statement PDFs
scripts/
  main.py         # raw-email dev CLI
  seed.py
  populate.py
```

### Request Lifecycle

1. FastAPI `lifespan` in `financial_dashboard.main` initializes the DB, starts support services, and stores a shared `FetchService` on `app.state`.
2. On each poll tick (or manual trigger), `FetchService.poll_all()` delegates to `integrations/email/orchestrator.py`.
3. HTML routes are split by domain under `web/` and aggregated by `web/__init__.py`.
4. Routes use `core.deps.get_session`, while supporting services live under `services/`.
5. During email processing the app:
    - Tries `_process_email()` (bank-email-parser).
    - On failure, tries `process_statement_email()` (cc-parser PDF path).
    - Saves `Email` row and, if successful, `Transaction` row.
    - Calls `link_transaction()` to set `account_id`/`card_id`.
6. JSON endpoints live under `api/`, while HTML routes render Jinja templates from `web/{domain}.py` and delegate to `services/`.

## Database Models

| Model | Table | Purpose |
|-------|-------|---------|
| `EmailSource` | `email_sources` | Gmail/Fastmail account with encrypted credentials and sync cursor |
| `FetchRule` | `fetch_rules` | Sender/subject/folder/bank match rule linked to a source |
| `Email` | `emails` | One row per fetched email; tracks parse status and links to a rule |
| `Transaction` | `transactions` | Parsed financial transaction; links to email, account, card |
| `Account` | `accounts` | Bank account, savings account, or credit card |
| `Card` | `cards` | Physical payment card linked to an account (supports addon cards) |
| `StatementUpload` | `statement_uploads` | CC statement PDF upload with reconciliation results stored as JSON |
| `BankStatementUpload` | `bank_statement_uploads` | Bank statement PDF upload with reconciliation results stored as JSON |
| `SmsMessage` | `sms_messages` | Incoming SMS alert with parse status and optional linked transaction |
| `PaisaExport` | `paisa_exports` | Paisa-mode export log for idempotent cross-channel dedup and journal regeneration |
| `Setting` | `settings` | Small key/value store for app-level settings |

### Schema Diagram

```mermaid
erDiagram
    EMAIL_SOURCES {
        int id PK
        string provider
        string label
        string account_identifier
        string credentials
        bool active
        string sync_cursor
        datetime last_synced_at
        string last_error
    }

    ACCOUNTS {
        int id PK
        string bank
        string label
        string type
        string account_number
        string statement_password
        string statement_password_hint
        bool active
    }

    CARDS {
        int id PK
        int account_id FK
        string card_mask
        string label
        bool is_primary
        bool active
    }

    FETCH_RULES {
        int id PK
        string provider
        int source_id FK
        string sender
        string subject
        string bank
        string folder
        string email_kind
        bool enabled
        datetime initial_backfill_done_at
    }

    EMAILS {
        int id PK
        string provider
        string message_id
        int source_id FK
        string remote_id
        string sender
        string subject
        datetime received_at
        datetime fetched_at
        string status
        text error
        int rule_id FK
    }

    STATEMENT_UPLOADS {
        int id PK
        int account_id FK
        int email_id FK
        string bank
        string filename
        string file_path
        string source_kind
        string status
        string card_number
        string statement_name
        string due_date
        string total_amount_due
        string minimum_amount_due
        int parsed_txn_count
        int matched_count
        int missing_count
        int imported_count
        text reconciliation_data
        text error
        datetime created_at
        string payment_status
        text payment_sent_offsets
        datetime payment_last_reminded_at
        decimal payment_paid_amount
        datetime payment_paid_at
    }

    BANK_STATEMENT_UPLOADS {
        int id PK
        int account_id FK
        int email_id FK
        string bank
        string filename
        string file_path
        string status
        string account_number
        string account_holder_name
        string opening_balance
        string closing_balance
        string statement_period_start
        string statement_period_end
        int parsed_txn_count
        int matched_count
        int missing_count
        int imported_count
        text reconciliation_data
        text error
        datetime created_at
    }

    SETTINGS {
        string key PK
        text value
        datetime updated_at
    }

    SMS_MESSAGES {
        int id PK
        string bank
        string sender
        text body
        datetime received_at
        datetime created_at
        string status
        int transaction_id FK
        text parse_error
        datetime parsed_at
    }

    PAISA_EXPORTS {
        int id PK
        string source
        int email_id FK
        int sms_message_id FK
        string idempotency_key UNIQUE
        string bank
        string email_type
        string direction
        decimal amount
        string currency
        date transaction_date
        time transaction_time
        string counterparty
        string reference_number
        string card_mask
        string account_mask
        string source_account
        string counterparty_account
        bool missing_account_mapping
        string status
        text error
        datetime created_at
        datetime updated_at
        datetime exported_at
    }

    TRANSACTIONS {
        int id PK
        int email_id FK
        int account_id FK
        int card_id FK
        int statement_upload_id FK
        int bank_statement_upload_id FK
        string bank
        string email_type
        string direction
        decimal amount
        string currency
        date transaction_date
        time transaction_time
        string counterparty
        string card_mask
        string account_mask
        string reference_number
        string channel
        decimal balance
        text raw_description
        text note
        string category
        datetime created_at
    }

    ACCOUNTS ||--o{ CARDS : has
    EMAIL_SOURCES ||--o{ FETCH_RULES : assigned_to
    EMAIL_SOURCES ||--o{ EMAILS : fetched_from
    FETCH_RULES ||--o{ EMAILS : matched_by
    ACCOUNTS ||--o{ STATEMENT_UPLOADS : owns
    EMAILS ||--o{ STATEMENT_UPLOADS : originates
    ACCOUNTS ||--o{ BANK_STATEMENT_UPLOADS : owns
    EMAILS ||--o{ BANK_STATEMENT_UPLOADS : originates
    EMAILS ||--o{ TRANSACTIONS : produces
    EMAILS ||--o{ PAISA_EXPORTS : produces
    SMS_MESSAGES ||--o{ PAISA_EXPORTS : produces
    SMS_MESSAGES ||--o{ TRANSACTIONS : source_sms
    TRANSACTIONS ||--o{ SMS_MESSAGES : linked_from
    ACCOUNTS ||--o{ TRANSACTIONS : linked_account
    CARDS ||--o{ TRANSACTIONS : linked_card
    STATEMENT_UPLOADS ||--o{ TRANSACTIONS : imported_from_cc_statement
    BANK_STATEMENT_UPLOADS ||--o{ TRANSACTIONS : imported_from_bank_statement
```

Relationship notes:
- `transactions.account_id` and `transactions.card_id` are the canonical account/card links after the linker runs; `card_mask` and `account_mask` remain as parser-derived denormalized hints.
- `settings` is intentionally standalone and has no foreign-key links.
- Several foreign keys are nullable (`fetch_rules.source_id`, `emails.source_id`, `emails.rule_id`, and the upload/transaction linkage columns), so rows can exist before the related record is known.

### Key Constraints
- `emails.message_id` is globally unique (prevents re-inserting the same email).
- `(source_id, remote_id)` is unique on `emails` (provider-scoped deduplication).
- `transactions` has a partial unique index on `(bank, reference_number)` where `reference_number IS NOT NULL` (deduplicates transactions with known UTR/UPI reference numbers).
- `(account_id, card_mask)` is unique on `cards`.
- `paisa_exports.idempotency_key` is unique.
- `paisa_exports` has lookup indexes on `(bank, direction, amount, currency, transaction_date)` and `(bank, direction, reference_number)`.

### Schema Migrations
There is no Alembic. Migrations are handled inline in `init_db()` via `try/except ALTER TABLE` blocks. A one-time migration removes a legacy `uq_transaction_dedup` constraint that was replaced by the partial index.

## Email Polling Detail

```
For each enabled FetchRule grouped by EmailSource:
  Open one provider connection (IMAP or JMAP session)
  For Gmail (IMAP):
    Phase 0: SEARCH by FROM/SUBJECT/SINCE -> collect UIDs per rule
    Phase 1: Batch FETCH headers + X-GM-MSGID -> deduplicate by X-GM-MSGID
    Phase 1.5: Bulk check UIDs against DB remote_ids -> filter to genuinely new
    Phase 2: FETCH RFC822 for new UIDs only (capped by fetch_limit)
  For Fastmail (JMAP):
    Email/query with filter -> collect (remote_id, blobId) per rule
    Bulk check remote_ids against DB -> filter to new
    Download blobs only for new emails
  For each new email:
    Extract metadata (sender, subject, date)
    Try bank-email-parser -> Transaction
    If fail: try cc-parser PDF path -> StatementUpload + Transactions
    Save Email row + Transaction row (if any)
    link_transaction() -> set account_id / card_id
  Mark initial_backfill_done_at on rules whose search completed
  Update source.last_synced_at
```

## CC Statement Processing Detail

```
During polling (automatic):
  Email subject contains "statement" AND has a PDF attachment?
    Yes: extract PDF bytes
    Try parsing without password
    If encrypted: try stored statement_password from all CC accounts for the bank
    If still can't parse: save PDF to data/statements/, create StatementUpload(status=password_required)
    If parsed:
      Find or create matching Account by card last-4
      reconcile_statement() -> matched / missing / extra
      Auto-import all missing as Transaction rows
      enrich_matched_transactions() -> write statement narration to DB counterparty where blank
      Create StatementUpload with full reconciliation JSON

Manual upload (/statements):
  User selects account and uploads PDF (with optional password)
  Same reconciliation flow
  User can select which missing transactions to import
  User can retry with password (and optionally save it to the account for future use)
```

## Seed Scripts

### scripts/seed.py
Seeds all default fetch rules for supported banks. Safe to run multiple times (idempotent — skips rules that already exist by `(provider, sender, subject)`). After adding email sources via the web UI, re-running `seed.py` will also auto-assign `source_id` to any unlinked rules that match by provider.

```bash
uv run python scripts/seed.py
```


## API Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/poll/status` | Returns current poll state, progress, last stats, last error |
| `POST` | `/api/transactions/{id}/note` | Update the freeform note on a transaction (JSON body: `{"note": "..."}`) |
| `POST` | `/api/sources/{id}/test` | Test connectivity for an email source; returns `{"ok": bool, "message"/"error": str}` |

All other routes are HTML (Jinja2 templates). Form submissions use POST + redirect (PRG pattern).

## Dev Tools

- `scripts/main.py` — CLI tool for listing/dumping emails from Gmail or Fastmail directly
- `scripts/populate.py` — Seed transactions from local `.eml` files in `data/`
- `scripts/seed.py` — Seed fetch rules for all known bank senders (idempotent)

## Related Projects

- **bank-email-parser** — Library that parses transaction alert emails from 12 Indian banks into structured data. Used as a dependency.
- **cc-parser** — Library that parses CC statement PDFs from 9 Indian banks. Used as a dependency for statement reconciliation.

## License

[MIT](LICENSE)
