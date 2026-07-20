# AGENTS.md — AI assistant instructions for financial-dashboard

## Project layout

```text
financial_dashboard/
  main.py                 FastAPI app factory + lifespan wiring
                           (bootstraps extensions before init_db runs)
  api/                    JSON routes
    extensions.py         /api/extensions + /api/extensions/paisa/*
  web/
    __init__.py           HTML router aggregation
    dashboard.py
    accounts.py
    sources.py
    rules.py
    transactions.py
    emails.py
    statements.py
    bank_statements.py    Included before statements.py
    settings.py
    polling.py
    forms.py
    networth.py           /networth + /networth/manual
    cas.py                /cas/upload
    extensions.py         /extensions + /extensions/paisa HTML surface (PRG)
  extensions/             First-party extension framework (manifest/registry)
    base.py               ExtensionManifest, Capability, ExtensionRegistrationError
    registry.py           ExtensionRegistry (ordered, rejects duplicate ids)
    paisa.py              PAISA_EXTENSION manifest + contributed paisa.* settings
    __init__.py           BUILTIN_EXTENSIONS + register_builtin_extensions()
  services/
    accounts.py
    emails.py
    fetch.py
    linker.py
    reminders.py
    rules.py
    settings.py
    sources.py
    telegram.py
    transactions.py
    extensions.py         ExtensionManager (app.state) + bootstrap_extensions()
     paisa/                Paisa integration: config/renderer/projection/
                            publisher/orchestrator (surface adapter for routes);
                            audit (ExtensionRun writer/query), automation
                            (PaisaAutomationRuntime: coalesced transaction-driven
                            sync coordinator + deduped failure notify),
                            report_cache (per-app TTL cache + coalescing),
                            reconciliation (read-only local/native/Paisa join),
                            diagnosis (sync-time Debit Entry fingerprint
                            classification vs. expected contra-expense postings)
    networth.py           current_networth + monthly_trend (forward-fill)
    snapshots.py          balance_snapshot upsert + emit helpers
    cas_ingestion.py      ingest_cas_payload (idempotent, NSDL-canonical)
    cas_emails.py         auto-fetch CAS emails via ensure_cas_fetch_rules
    manual_items.py
    statements/
      __init__.py
      cc.py
      bank.py
      shared.py
      dates.py
  integrations/
    parsers.py
    email/
      base.py
      body.py
      parsing.py
      imap_gmail.py
      jmap_fastmail.py
      orchestrator.py
  core/                   Shared templating, date, auth, crypto, deps helpers
  db/                     Engine/session setup, models, enums, init_db glue
  templates/              Jinja templates
  static/                 CSS and JS assets
  data/
    failed/               Failed parse spools (.eml files, max 7 days old)
    statements/           Saved statement PDFs

scripts/
  main.py                 Raw-email dev CLI
  seed.py
  populate.py
  synth/                  Deterministic synthetic seed + offline Paisa 0.7.4
                           corpus generator (generate/load/verify/reset)
  paisa_contract.py       Optional Docker probe against ananthakumaran/paisa:v0.7.4
```

## How to run

```bash
uv run fastapi dev
uv run python scripts/seed.py
uv run pytest -q
# optional, dev-only — never touches a production DB:
uv run python -m scripts.synth generate
uv run python scripts/paisa_contract.py --skip-if-unavailable   # needs Docker
```

## Service conventions

### Session handling

- Route handlers get `session: AsyncSession = Depends(get_session)` from `core/deps.py`.
- Services take `session: AsyncSession` as a required first parameter when the caller already owns the request session.
- Background tasks (fetch polling, Telegram handlers, reminders) open their own session with `async with async_session() as session:` and pass it onward where applicable.
- Do not add `async_session_factory` fallback parameters.

## Extension conventions

- **No plugin discovery.** Extensions are registered explicitly from
  `financial_dashboard/extensions/` into an `ExtensionRegistry`. The set of
  extensions is exactly `BUILTIN_EXTENSIONS` (today: Paisa only).
- **Bootstrap ordering.** `register_builtin_extensions()` runs in the lifespan
  *before* `init_db()` calls `load_all_settings()`, so contributed `SettingDef`
  entries are present when the settings cache fills. Setting registration is
  idempotent: an equal re-defn is accepted, a conflicting one raises
  `ExtensionRegistrationError`.
- **Paisa never mutates core rows.** Projection is a one-way read of
  `accounts`/`transactions` into a generated ledger include file; generate/sync
  never writes `accounts`, `transactions`, etc. Keep it that way.
- **Failure isolation is mandatory.** Probe/preview/generate/sync dispatch
  through `services.paisa.surface`, which catches core errors into typed
  `ok=false` responses (JSON) or `?error=`/`?outcome=` flashes (HTML). An
  optional-extension failure must never 500 or affect a core route.
- **Flash values are URL-encoded.** `web/extensions.py:_flash` runs both key and
  value through `urllib.parse.urlencode` so spaces, `&`, `#`, or Unicode cannot
  corrupt the redirect `Location`. Don't reintroduce raw string concatenation
  into a flash query.
- **Status badge is DOM-built, never `innerHTML`.** The Paisa status fetch
  builds the dot via `createElement`/`classList`/`replaceChildren` + a text
  node, and the detail line via `textContent`, so a hostile/misconfigured
  upstream string cannot inject markup. The mapping-row helper's `innerHTML` is
  intentional (hardcoded constants only) — leave it.
- **Modes are `disabled` / `connect` / `project`.** `connect` is read-only
  (probe only); only `project` may generate/sync. The orchestrator enforces
  this; don't bypass it. Curated report reads (`report_summary`) require
  `can_connect`; `disabled` makes zero upstream calls.
- **Curated reports are typed, never raw-proxied.** `integrations/paisa.py`
  owns one normalizer per upstream v0.7.4 endpoint (budget/allocation/recurring/
  income_statement/liabilities/assets_balance) producing dashboard-owned
  NamedTuples; `surface._report_to_dto` is the only place those meet the schema
  Pydantic DTOs. A server-side TTL cache
  (`services/paisa/report_cache.py:PaisaReportCache`) lives on
  `app.state.paisa_report_cache` (per-app, never a module global), coalesces
  concurrent same-key reads into one upstream call, and is bounded in entry
  count. Report/reconciliation pages render client-side so a slow/down Paisa
  never blocks page render; all values are DOM-built (`textContent`/`createElement`).
- **Manual operations are audited.** Manual generate/sync/probe dispatch through
  `surface.*_audited`, which records a start/complete `ExtensionRun` row on the
  request session and commits. `details` carries only sanitized counts/hashes —
  never credentials or raw journal text.
- **Sync diagnosis classifies expected contra-expense postings.** Paisa v0.7.4's
  doctor emits a `Debit Entry` danger for *every* negative `Expenses:` posting,
  which collides with our canonical contra-expense semantics (refund/cashback/
  reversal postings are negative Expenses so they net). Paisa exposes no config
  flag to disable/allowlist these checks, so `services/paisa/diagnosis.py`
  derives a multiset of expected `(account, date, amount)` fingerprints from the
  generated `ProjectionReport` and accepts only a `Debit Entry` danger whose
  parsed fingerprint exactly consumes one (multiplicity-aware). An unmatched
  `Debit Entry`, an unparseable issue, and every other danger kind
  (`Negative Balance`, `Credit Entry`, …) stay fatal. Keep it that way: never
  blanket-suppress diagnosis summaries, never remap refunds/cashback to Income,
  and never touch the probe path (it surfaces the raw upstream diagnosis). The
  sync DTO/audit expose additive `diagnosis_expected`/`diagnosis_accepted`/
  `diagnosis_fatal` counts — no raw journal text or credentials.
- **Reconciliation is read-only and joins only by explicit mappings.**
  `services/paisa/reconciliation.py` never writes a core row, never corrects a
  balance, and joins native ↔ Paisa only through `paisa.account_mappings` (exact
  match or direct-child rollup — no fuzzy matching). Mapping suggestions are
  preview-only defaults that must be accepted via the normal config-save path.
- **Auto sync is a coalesced, transaction-driven coordinator.** With
  `paisa.auto_sync_enabled=true` and `project` mode, a coordinator polls
  `extension_sync_state` (every 2s) and reconciles whenever
  `desired_revision > applied_revision` (or `force_reload`). SQLite AFTER
  triggers on `transactions`/`accounts`/`cards`/`balance_snapshots`/
  `investment_lots`/`cas_uploads` (insert/update/delete) and on `settings`
  (`paisa.%` keys only) bump `desired_revision` with **exact post-commit
  semantics** — the bump shares the dirtying write's transaction, so it commits
  or rolls back together (a rolled-back savepoint drops only its own bump); a
  coordinator never observes a revision for an uncommitted change. They also
  maintain `first_dirty_at`/`last_dirty_at`; a `paisa.%` settings change
  additionally resets the retry backoff. No triggers exist on
  `extension_sync_state`/`extension_runs` (recursion guard) or on tables the
  projection does not read. Fixed, non-tunable timings: 5s quiet debounce, 30s
  max dirty latency, 1/2/5/10/15-min retry backoff, six-hour force reload. Each
  reconcile is a **full-journal** Paisa `/api/sync` (no partial-transaction API
  exists), so a bulk statement import (one outer commit, ~200 rows) is one bump
  and one coalesced reload. `disabled`/`connect` accumulate dirty state with no
  I/O. `paisa.auto_sync_min_interval_minutes` (default **1**, persisted values
  preserved) is a hard floor between remote reloads/retries only — **not** the
  event debounce or max latency. Keep the triggers post-commit-transactional;
  never relax exact-post-commit, never make the debounce/latency/backoff
  operator-tunable, and never introduce a partial-transaction sync path.
- **Auto-sync failure notifications are deduped.** When
  `paisa.notify_sync_failures=true` and an automatic reconcile fails, the
  coordinator calls the existing Telegram service (best-effort, isolated).
  Repeated *identical* failures (same outcome + sanitized error) are deduped
  via a fingerprint persisted in the audit `details` (`notify_fp`) so the
  dedupe survives restarts; a changed failure notifies again. Notification never
  runs when Telegram is unconfigured, and a notification failure never affects
  reconcile or audit.
- **Synthetic tooling never touches prod.** `scripts/synth/` writes only to a
  dedicated `data/synthetic/<profile>/synthetic.db` guarded by path checks;
  `scripts/paisa_contract.py` is an optional, Docker-only developer probe. The
  runtime never spawns Paisa.
- **Dashboard taxonomy semantics, not direction-only.** The projection roots
  the contra account by *category* (income→Income, expense→Expenses always so
  reversals net, refund/cashback→contra-expense, investment→
  `Assets:Investments:Unallocated` asset movement, repayment→
  `Equity:Transfers In` non-income clearing). Direction affects the sign only.
  `self_transfer`/`credit_card_payment` stay special-cased above this table.
  Operator `category_mappings` overrides always win. `emi_loan`/
  `cash_withdrawal` are imprecise (conservative Expenses clearing + diagnostic;
  no fabricated principal/cash accounts).
- **Card payment resolution is exact-match only.** A bank-side
  `credit_card_payment` resolves to a specific selected liability only by
  explicit `card_id` or exact `card_mask` — never fuzzy. Unresolved posts to
  `Liabilities:Credit Card` with `dashboard_card_resolution=unresolved`.
- **Investment funding dedup is conservative.** When lot projection is on and
  a bank investment txn provably funds a lot (exact reference that maps to a
  single instrument, or deterministic exact date+amount), the bank leg's contra
  is remapped to `Equity:Opening Balances:Investment` so the asset is counted
  once. A reference shared by multiple instruments **does not early-abort** —
  it falls through to the deterministic exact date+amount check, which may
  still disambiguate to a single instrument and remap (otherwise suppress
  conservatively). If the link is potential but not provably deterministic
  (shared reference with no deterministic date+amount, or a date-only/
  amount-only collision), the matching lots are suppressed (never emit both)
  and any suppressed instrument's price directive is dropped so no orphan
  `P`/`price` line lingers.
- **Canonical `dashboard_*` metadata, two schemas.** Transaction-derived
  entries (standard/reversal/FX/self-transfer/card-payment) carry the full
  schema (`dashboard_txn_ids`, `dashboard_kind`, `dashboard_category`,
  `dashboard_source`, `dashboard_channel`, `dashboard_email_type`,
  `dashboard_account_ids`, `dashboard_card_ids`, sanitized
  `dashboard_reference`). Source-less entries carry a *reduced* schema —
  openings carry posting-level `dashboard_account_ids`/`dashboard_source`/
  `dashboard_as_of`, lots carry `dashboard_kind`/`dashboard_instrument`/
  `dashboard_acquired_on` plus optional CAS provenance. Do not claim
  transaction-only fields (`dashboard_txn_ids`/`dashboard_category`/
  `dashboard_channel`/`dashboard_email_type`/`dashboard_card_ids`) on openings
  or lots — they have no dashboard Transaction. Ledger/hledger: one
  `; key: value` tag per line (space after colon, no comma). Beancount:
  lowercase `key: "value"`. Backward-compatible `txn: <id>` tag preserved on
  transaction-derived entries only. Values sanitized via
  `sanitize_meta_value` — no secrets/raw bodies/full masks. Validated through
  real beancount `loader.load_string` meta identity (standard, opening, lot,
  resolved/unresolved card-payment and FX entries — not substrings) and
  ledger/hledger tag queries where the binaries are available.

## Style conventions

- **No `from __future__ import annotations`.** Python 3.14 + PEP 649
  makes lazy annotation evaluation the default, so the future import is
  a no-op. Don't add it to new files; existing files have it stripped.
- **Multi-value returns use `typing.NamedTuple`**, not anonymous
  tuples and not frozen dataclasses (for the typical 2–4-field case).
  NamedTuple keeps positional unpacking (`a, b = func()` and `result[0]`)
  working at every call site while giving named attribute access
  (`result.field`) for readability. Reserve `@dataclass(frozen=True)`
  for value objects with methods or many fields; reserve `BaseModel`
  for things that cross an API boundary (already the convention in
  `schemas/`). Examples in the repo: `CasEmailProcessResult`,
  `FetchSourceResult`, `MergeTransactionResult`, `ProcessedEmailParse`,
  `PdfAttachment`, `SmsIngestResult`, `RawEmailLoadResult`.
- **No defensive `getattr(obj, "attr", default)`** for ORM columns or
  any attribute that is always present on the typed object. Use direct
  attribute access; let attribute errors surface. The only legitimate
  uses are: `request.app.state.<attr>` (state attrs may be unset before
  startup), dynamic-name `getattr(obj, field_name_variable)`, and
  module-level `def __getattr__` for lazy imports.

## Compatibility rules

- Preserve current HTTP routes, JSON response shapes, template behavior, parser-derived `email_type` values, and script entrypoints unless a task explicitly allows a breaking change.
- Shared poll state belongs on `app.state.fetch_service`; avoid new module-level poll loops or duplicate status dicts.
- Keep `bank_statements.router` registered before `statements.router`.

## Database schema docs

- The Mermaid ER diagram in `README.md` is documentation for `financial_dashboard/db/models.py`; when models change, update the diagram, the model summary table, and the key-constraints notes together.
- Keep every ORM table in the diagram, including standalone tables such as `settings`, and include every real foreign-key edge from `cards`, `fetch_rules`, `emails`, `statement_uploads`, `bank_statement_uploads`, and `transactions`.
- If a field is only a parser/linker hint (`card_mask`, `account_mask`, similar denormalized values), describe it in nearby prose instead of drawing a fake foreign-key relationship.

## Local cross-repo development

**The committed state always uses git/PyPI-tagged sibling versions.**
`pyproject.toml`'s `[tool.uv.sources]` pins each sibling to its
`git = "https://github.com/..."` URL, and `uv.lock` records the exact SHA
that CI, deploys, and reviewers see.

When you're actively developing a change that spans this repo and a
sibling (`bank-email-parser`, `bank-statement-parser`, `cc-parser`,
`cas-parser`), you can **temporarily** swap those git sources for local
path sources so edits in the sibling are picked up immediately — no
reinstall, no version bump, no lockfile churn. The siblings live at
`../<name>` relative to this repo.

Temporary dev-only block (do NOT commit this):

```toml
[tool.uv.sources]
bank-email-parser = { path = "../bank-email-parser", editable = true }
bank-statement-parser = { path = "../bank-statement-parser", editable = true }
cc-parser = { path = "../cc-parser", editable = true }
cas-parser = { path = "../cas-parser", editable = true }
```

Rules:

- **Before pushing or opening a PR**: revert `[tool.uv.sources]` to the
  `git = "..."` form and run `uv lock` so the lockfile pins the new SHA
  (tag the sibling repo first if the change isn't already on `main`).
  Deploys run off these git sources; path sources would break CI.
- **Never commit path sources.** If `git diff pyproject.toml uv.lock` shows
  path entries or a stale lockfile, fix that before pushing.
- Sibling repo SHAs pinned by the committed lockfile can be inspected in
  `uv.lock` under the relevant `[[package]]` block.

## Quality gates

Run all of these before finishing a refactor:

- `uv run ruff check financial_dashboard tests scripts`
- `uv run ruff format --check financial_dashboard tests scripts`
- `uv run ty check financial_dashboard`
- `uv run pytest -q`

Use `uv run` for every command. PEP 758 parenthesis-free `except X, Y:` syntax is valid in this repo. Prefer `python-dateutil` helpers from `core/dates.py`.
